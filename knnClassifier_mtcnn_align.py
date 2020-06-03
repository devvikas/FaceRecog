"""An example of how to use your own dataset to train a classifier that recognizes people.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

class FaceAligner:
    def __init__(self, desiredLeftEye=(0.27, 0.27), desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, keypoints):

        # convert the landmark (x, y)-coordinates to a NumPy array
        # print('keypoints >> ', keypoints)                    
        leftEyePts = np.asarray([keypoints['right_eye']])
        rightEyePts = np.asarray([keypoints['left_eye']])
        
        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye

        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        
        
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        
        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output

def normalize_image(image):    

    [B,G,R] = np.dsplit(image, image.shape[-1])

    Rx = (R - 122.782) / 256.
    Gx = (G - 117.001) / 256.
    Bx = (B - 104.298) / 256.

    new_image = np.dstack((Rx, Gx, Bx))
    return new_image


class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    nrof_classes = 15
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

from converter.model import ScaleLayer, ReshapeLayer
# from scipy.spatial import distance
import cv2, os, math
import imutils
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

# PATH_TO_DLIB_H5_MODEL = 'checkpoints/3105/facenet_10.h5'
PATH_TO_DLIB_H5_MODEL = 'model/dlib_face_recognition_resnet_model_v1.h5'
# predictor_path = 'shape_predictor_5_face_landmarks.dat'
model_from_file = tf.keras.models.load_model(PATH_TO_DLIB_H5_MODEL, custom_objects={'ScaleLayer': ScaleLayer, 'ReshapeLayer': ReshapeLayer})
fa = FaceAligner(desiredFaceWidth=150)

#TRAIN ../data/images/train_celeb_align/ ../models/mobilenet/MobileFaceNet_9925_9680.pb ../models/mobilenet_knn_classifier_celeb5.pkl
#CLASSIFY ../data/images/val50_celeb_align/ ../models/mobilenet/MobileFaceNet_9925_9680.pb ../models/mobilenet_knn_classifier_celeb5.pkl

def main(args):
    # with tf.Graph().as_default():

        # with tf.Session() as sess:

            np.random.seed(seed=args.seed)

            if args.use_split_dataset:
                dataset_tmp = get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class,
                                                    args.nrof_train_images_per_class)
                if (args.mode == 'TRAIN'):
                    dataset = train_set
                elif (args.mode == 'CLASSIFY'):
                    dataset = test_set
            else:
                dataset = get_dataset(args.data_dir)
            
            # print(dataset)
            # Check that there are at least one training image per class
            for cls in dataset:
                assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

            paths, labels = get_image_paths_and_labels(dataset)
            # print(labels)
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            # Load the model
            print('Loading feature extraction model')

            embedding_size  = 128
            # Run forward pass to calculate embeddings
            
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            # print(emb_array.shape)
            print(nrof_batches_per_epoch)
            
            for i in range(nrof_batches_per_epoch):

                start_index = i 
                end_index = i + 1
                paths_batch = paths[start_index:end_index]
                print(paths_batch[0])
                image2 = cv2.imread(paths_batch[0])
                if image2 is None:
                    continue
                
                (bg_h, bg_w) = image2.shape[:2]
                result = detector.detect_faces(image2)
                print( result)
                if len(result) == 0:
                    print('-------------faceAligned REMVIN')
                    os.remove(paths_batch[0])
                    continue

                         
                keypoints = result[0]['keypoints']    

                faceAligned = fa.align(image2, keypoints)
                # cv2.imshow('faceAligned', faceAligned)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                if faceAligned is None:
                    print('-------------faceAligned REMVIN')
                    os.remove(paths_batch[0])
                    continue
                
                norm_img = normalize_image(faceAligned.copy())
                image2 = cv2.resize(norm_img, (150, 150))
#                 # cv2.imshow('s', img)
#                 # cv2.waitKey(0)
#                 # cv2.destroyAllWindows()
                    
                expanded_image = np.expand_dims(image2, axis=0)
                pred = model_from_file.predict(expanded_image)
                emb_array[start_index:end_index, :] = pred

            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if (args.mode == 'TRAIN'):
                # Train classifier
                print('Training classifier')
                model = KNeighborsClassifier(n_neighbors=3)
                print(emb_array)
                model.fit(emb_array, labels)
                # Create KNN classifier
                # Create a list of class names
                class_names = [cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

                # print('Loaded classifier model from file "%s"' % classifier_filename_exp)
                # with open('../models/mob_embed.pkl', 'wb') as outfile:
                #     pickle.dump((emb_array, class_names), outfile)

            elif (args.mode == 'CLASSIFY'):
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                #with open('../models/embed.pkl', 'rb') as infile:
                #    (known_emb_array, known_labels) = pickle.load(infile)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                create_confusion_matrix(best_class_indices,labels,class_names)

                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)


def create_confusion_matrix(predicted_classes,ground_truth,class_names):
    f = open('knnresults_mobilenet_may5_2_full.csv', 'w')
    csv_writer = csv.writer(f, delimiter=',')
    nclasses = len(class_names)
    cf = np.zeros((nclasses, nclasses), dtype='int64')
    for i, j in zip(ground_truth, predicted_classes.tolist()):
        cf[i][j] += 1
    # pprint(cf)
    labels = class_names
    ac = sum(np.diag(cf)) / np.sum(cf)
    csv_writer.writerow(['Accuracy'] + [round(ac * 100, 2)])
    csv_writer.writerow([''])
    csv_writer.writerow([''])
    csv_writer.writerow([' '] + labels)
    for i in range(len(cf)):
        print(cf[i])
        # [labels[i]].extend(cf[i])
        out = [labels[i]] + list(cf[i])
        csv_writer.writerow(out)

    true_pos = np.diag(cf)
    false_pos = np.sum(cf, axis=0) - true_pos
    false_neg = np.sum(cf, axis=1) - true_pos

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    # prec = np.where(np.isnan(precision), 0, precision)
    # prec = np.sum(prec)/np.count_nonzero(prec)
    # rec = np.where(np.isnan(recall), 0, recall)
    # rec = np.sum(rec)/np.count_nonzero(rec)
    # precision = np.append(precision,[prec])
    # recall = np.append(recall,[rec])
    precs = np.round(precision * 100, 2)
    recs = np.round(recall * 100, 2)

    csv_writer.writerow([''])
    csv_writer.writerow([''])
    csv_writer.writerow(['Precsion'] + list(precs))
    csv_writer.writerow(['Recall'] + list(recs))
    f.close()

def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths) >= min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
                        help='Indicates if a new classifier should be trained or a classification ' +
                             'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename',
                        help='Classifier model file name as a pickle (.pkl) file. ' +
                             'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset',
                        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +
                             'Otherwise a separate test set can be specified using the test_data_dir option.',
                        action='store_true')
    parser.add_argument('--test_data_dir', type=str,
                        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=1)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=150)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
                        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
                        help='Use this number of images from each class for training and the rest for testing',
                        default=10)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

# TRAIN  train_celeb/  model/dlib_face_recognition_resnet_model_v1.h5  knn_classifier_celeb5_may31.pkl
# CLASSIFY ../../../data/val_celeb50/ dlib_face_recognition_resnet_model_v1.h5  knn_classifier_celeb5_may31.pkl
    