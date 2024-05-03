import os
import sys
import time

import torch

import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

from pymongo import MongoClient
from facenet_pytorch import * 
import faiss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
documents = []
KEEP_ALL = False

# create MongoDB connection
connection = "mongodb+srv://duongminhhieu1611:Cuncon1998.@cluster0.w7bjhm4.mongodb.net/"

client = MongoClient(connection)

database = 'facenet'; collection = 'facenet'
db = client[database]

###### retrieve data from mongoDB
time0 = time.time()
data = db[collection].find()


def build_detector(detector_name="mtcnn", net_type="slim", size="320"):
    if detector_name == "mtcnn":
        
        detector = MTCNN(keep_all=KEEP_ALL, device=device)
    
    elif detector_name == "opencv":
        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    elif detector_name == "ssd":
        
        input_img_size = int(size)
        define_img_size(input_img_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'


        
        label_path = "./facenet_pytorch/models/vision/voc-model-labels.txt"
        net_type = "RFB" # RFB or slim
        class_names = [name.strip() for name in open(label_path).readlines()]
        num_classes = len(class_names)
        model_path = "./facenet_pytorch/models/vision/pretrained/version-" + net_type + "-" + size + ".pth"
        if net_type == 'slim':
            net = create_mb_tiny_fd(len(class_names), is_test=True, device=device)
            net.load(model_path)
            detector = create_mb_tiny_fd_predictor(net, candidate_size=1000, device=device)
        elif net_type == 'RFB':
            net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=device)
            net.load(model_path)
            detector = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=1000, device=device)
            print(detector)
        else:
            print("The net type is wrong!")
            sys.exit(1)

    return detector

def build_recognizer(recognizer_name="facenet"):
    if recognizer_name == "facenet":
        recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    # elif recognizer_name == "mobilefacenet":
        
    
    return recognizer

def create_database(db_path=".", detector_name="mtcnn", net_type="slim", size="320", recognizer_name="facenet", update=True):
    # build detector and recognizer
    detector = build_detector(detector_name=detector_name, net_type=net_type, size=size)
    recognizer = build_recognizer(recognizer_name=recognizer_name)
    facial_img_paths = []
    # read image paths
    if os.path.isdir(db_path):
        labels = os.listdir(db_path)
        
        folders =  [os.path.join(db_path, label) for label in labels]
        for folder in folders:
            for img_label in os.listdir(folder):
                facial_img_paths.append(os.path.join(folder, img_label))
    else:
        facial_img_paths.append(db_path)
    # read and process images
  
    instances = []

    for i in tqdm(range(0, len(facial_img_paths))):
        path = facial_img_paths[i]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # get largest cropped image tensor
        img_cropped = detector(img)[0][0]
        # calculate embedding (unsqueeze to add batch dimension)
        embedding = recognizer(img_cropped.unsqueeze(0).to(device))[0].detach().cpu()
        
        instance = []
        instance.append(path)
        instance.append(embedding)
        instance.append(path.split("/")[-2])
        instances.append(instance)


    df = pd.DataFrame(instances, columns = ["img_name", "embedding", "label"])
    # print(df.head())

    # store embeddings in mongoDb
    if update:
        for index, instance in tqdm(df.iterrows(), total=df.shape[0]):
            db[collection].update_one({"img_path": instance["img_name"]},{"$set":{"embedding": instance["embedding"].tolist(), "id": instance["label"]}}, upsert=True)
    else:
        for index, instance in tqdm(df.iterrows(), total=df.shape[0]):
            db[collection].insert_one({"img_path": instance["img_name"]}, {"embedding": instance["embedding"].tolist(), "id": instance["label"]})

def find_euclidean_distance(source_representation=None, test_representation=None):
    """
    Find euclidean distance between two given vectors
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated euclidean distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.array(euclidean_distance)
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def createFaissDB():
    ###### FAISS

    index = faiss.IndexFlatL2(512)
    representations = []
    for doc in documents:
        d = []
        d.append(doc["id"])
        d.append(doc["embedding"])
        
        representations.append(d)
    embeddings = []
    for i in range(0, len(representations)):
        embedding = representations[i][1]
        embeddings.append(embedding)
    embeddings = np.array(embeddings, dtype='f')
    index.add(embeddings)
    return representations, index

def detect_faces(img=None, detector_name="mtcnn", detector=None):
    if detector_name == "mtcnn":
        faces, boxes = detector(img)
    elif detector_name == "opencv":
        boxes_ = detector.detectMultiScale(
            img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        
        faces = []
        boxes = []
        for (x, y, w, h) in boxes_:
            x1, y1, x2, y2 = x, y, x+w, y+h
            boxes.append((x1, y1, x2, y2))
        faces_im = []
        for box in boxes:
            face = extract_face(img, box)
            face = fixed_image_standardization(face)
            faces_im.append(face)
            if KEEP_ALL is False:
                break

    elif detector_name == "ssd":
        boxes, _, _ = detector.predict(img, 100, prob_threshold = 0.7)
        faces = []
        # print(boxes)
        faces_im = []
        for box in boxes:
            face = extract_face(img, box)
            face = fixed_image_standardization(face)
            faces_im.append(face)
            if KEEP_ALL is False:
                break
    if detector_name == "ssd" or detector_name == "opencv":
        if len(faces_im) > 0:
            if KEEP_ALL:
                faces = torch.stack(faces_im) 
            else:
                faces = torch.stack(faces_im[:1])
            
    return faces, boxes

def process_image(frame=None, mode="faiss", detector_name="mtcnn", detector=None, recognizer=None):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if mode == "faiss":
        
        ###### FAISS
        representations, index = createFaissDB()

    targets = []
    
    faces, boxes = detect_faces(img=img, detector_name=detector_name, detector=detector)
 
            
    if faces is not None and boxes is not None:
        for (face, box) in zip(faces, boxes):
            
            target = []
            target_embedding = recognizer(face.unsqueeze(0).to(device))[0].detach().cpu()

            if mode == "faiss":
                ###### FAISS
                target_embedding = np.array(target_embedding)
                target_embedding = np.expand_dims(target_embedding, axis=0)

                f_dists, f_ids = index.search(target_embedding, k=2)
                idx = f_ids[0][0]
                id = representations[idx][0]
                
                target.append(id)
                target.append(box)
                targets.append(target)
            elif mode == "mongodb":
                ###### MONGODB
                query = db[collection].aggregate([
                    {
                        "$set": {
                            "target_embedding": target_embedding.tolist()
                        }
                    }
                    , {"$unwind": {"path": "$embedding", "includeArrayIndex": "embedding_index"}}
                    , {"$unwind": {"path": "$target_embedding", "includeArrayIndex": "target_index"}}
                    , {
                        "$project": {
                            "img_path": 1,
                            "embedding": 1,
                            "target_embedding": 1,
                            "compare": {
                                "$cmp": ['$embedding_index', '$target_index']
                            }
                        }
                    }
                    , {"$match": {"compare": 0}}
                    , {
                        "$group": {
                            "_id": "$img_path",
                            "distance": {
                                "$sum": {
                                    "$pow": [{
                                        "$subtract": ["$embedding", "$target_embedding"]
                                    }, 2]
                                }
                            }
                        }
                    }
                    , {
                        "$project": {
                            "_id": 1
                            , "distance": {"$sqrt": "$distance"}
                            , "cond": {"$lte": ["$distance", 10]}
                        }
                    }
                    , {"$match": {"cond": True}}
                    , {"$sort": {"distance": 1}}
                ])
                
                for i in query:
                    id = i["_id"].split("/")[-2]

                    break
            elif mode == "local":
                # find distance locally (laptop)
                dists = []
                for doc in documents:
                    d = []
                    dist = find_euclidean_distance(target_embedding, doc["embedding"])
                    d.append(doc["id"])
                    d.append(dist)
                    dists.append(d)

                dists.sort(key=lambda x: x[1])
                print(dists)
                target.append(dists[0][0])
                target.append(box)
                targets.append(target)
            
    # Draw bounding boxes
    if faces is not None and boxes is not None:
        for target in targets:
            id, (x1, y1, x2, y2) = target
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, str(id), (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    
    cv2.imshow("Face recognition", frame)

def main():
    
    img = "./aaa.jpg"
    # create mongoDB database
    create_database()

    # recognize image
    process_image(img=img)
    
def stream(mode="faiss", detector_name="mtcnn", net_type="slim", size="320", recognizer_name="facenet"):

    # build detector and recognizer
    detector = build_detector(detector_name=detector_name, net_type=net_type, size=size)
    recognizer = build_recognizer(recognizer_name=recognizer_name)

    # capture images from camera
    video_capture = cv2.VideoCapture(0)
    # used to record the time when we processed last frame 
    prev_frame_time = 0
    
    # used to record the time at which we processed current frame 
    new_frame_time = 0
    while True:
        time0 = time.time()
            
        result, frame = video_capture.read()
        if result is False:
            break
        # time when we finish processing for this frame 
        new_frame_time = time.time() 
        # fps will be number of frame processed in given time frame 
        # since their will be most of time error of 0.001 second 
        # we will be subtracting it to get more accurate result 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
    
        # converting the fps into integer 
        fps = int(fps) 
    
        # converting the fps to string so that we can display it on frame 
        # by using putText function 
        fps = str(fps) 
    
        # putting the FPS count on the frame 
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA) 

        # recognize face(s)
        
        process_image(frame, mode=mode, detector_name = detector_name, detector=detector, recognizer=recognizer)
        print(time.time()-time0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # main()
    # db_path = "./facenet_pytorch/data/test_images"
    # create_database(db_path=db_path, detector_name="mtcnn", net_type="slim", size="320", recognizer_name="facenet")
    ###### read data from database
    
    for datum in data:
        documents.append(datum)
    print("Retrieved data completed. Time: " + str(time.time() - time0))
    stream(mode="faiss", detector_name="ssd", net_type="slim", size="320", recognizer_name="facenet")