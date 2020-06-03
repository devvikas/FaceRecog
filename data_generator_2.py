# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:41:01 2020

@author: vikas
"""

import cv2, os, pickle, random
import numpy as np
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
import tensorflow as tf
import imgaug.augmenters as iaa

# scale=(0, 5)
# aug1 = iaa.Add((0, 20))
# aug2 = iaa.AddElementwise((0, 20), per_channel=0.5)
# aug3 = aug = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, .1))

# aug4 = iaa.AdditiveLaplaceNoise(scale=(0, 0.01*255))
# aug5 = iaa.AdditivePoissonNoise(scale)
# aug6 = iaa.Multiply((0.5, 1.0), per_channel=0.5)

# aug7 = iaa.Cutout(nb_iterations=1, size=0.01)
# aug8 = iaa.Cutout(fill_mode="constant", size=0.01, cval=10)
# aug9 = iaa.Cutout(fill_mode="gaussian", fill_per_channel=True, size=0.05)

# aug10 = iaa.Dropout(p=(0, 0.1))
# aug11 = iaa.CoarseDropout((0.0, 0.1), size_percent=(0.02, 0.1))
# aug12 = iaa.Dropout2d(p=0.1, nb_keep_channels=0)
# aug13 = iaa.ImpulseNoise(0.1)
# aug14 = iaa.CoarseSaltAndPepper(0.1, size_percent=(0.01, 0.02))
# aug15 = iaa.CoarsePepper(0.1, size_percent=(0.01, 0.02))
# # aug16 = iaa.Invert(0.25, per_channel=0.4)
# # aug17 = iaa.Invert(0.1)
# aug18 = iaa.Solarize(0.1, threshold=(1, 32))
# # aug19 = iaa.JpegCompression(compression=(95, 99))
# aug20 = iaa.GaussianBlur(sigma=(0.0, 2.0)) # blur
# aug21 = iaa.AverageBlur(k=((1, 3), (1, 3)))

# aug22 = iaa.MotionBlur(k=3)

# aug23 = iaa.BlendAlpha(
#     (0.0, 0.5),
#     iaa.Affine(rotate=(0, 0)),
#     per_channel=0.5)

# aug24 = iaa.BlendAlpha(
#     (0.0, 0.5),
#     foreground=iaa.Add(20),
#     background=iaa.Multiply(0.1))

# aug25 = iaa.BlendAlphaMask(
#     iaa.InvertMaskGen(0.3, iaa.VerticalLinearGradientMaskGen()),
#     iaa.Clouds()
# )

# aug26 = iaa.BlendAlphaElementwise(
#     (0.0, 0.5),
#     iaa.Affine(rotate=(-20, 20)),
#     per_channel=0.5)

# aug27 = iaa.BlendAlphaElementwise(
#     (0.0, 0.2),
#     foreground=iaa.Add(20),
#     background=iaa.Multiply(0.4))
# aug28 = iaa.BlendAlphaElementwise([0.25, 0.75], iaa.MedianBlur(13))

# aug29 = iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(0.05))
# aug30 = iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(0.05), upscale_method="nearest")
# aug31 = iaa.Cutout(fill_mode="constant", cval=(0, 40), fill_per_channel=0.2)
# aug32 = iaa.BlendAlphaHorizontalLinearGradient( iaa.TotalDropout(0.01), min_value=0.01, max_value=0.02)


# aug33 = iaa.BlendAlphaHorizontalLinearGradient(
#     iaa.AveragePooling(3),
#     start_at=(0.0, 0.1), end_at=(0.0, 0.2))

# aug34 = iaa.BlendAlphaVerticalLinearGradient(
#     iaa.TotalDropout(0.01),
#     min_value=0.01, max_value=0.2)

# aug35 = iaa.BlendAlphaVerticalLinearGradient(
#     iaa.AveragePooling(3),
#     start_at=(0.0, 0.1), end_at=(0.0, 0.2))

# aug36 = iaa.BlendAlpha(
#     (0.0, 0.3),
#     iaa.Affine(rotate=(0, 0)),
#     per_channel=0.3)

# aug38 =  iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB", children=iaa.WithChannels(0,iaa.Add((0, 10))))

# aug40 = iaa.WithHueAndSaturation(iaa.WithChannels(0, iaa.Add((0, 10))))
# aug41 = iaa.MultiplyHueAndSaturation((0.1, .9), per_channel=True)
# aug42 = iaa.AddToHueAndSaturation((0, 4), per_channel=True)
# aug43 = iaa.AddToHue((0, 4))
# aug44 = iaa.AddToSaturation((0, 4))
# aug45 = iaa.Sequential([ iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
#     iaa.WithChannels(0, iaa.Add((0, 10))),
#     iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")])
    
# aug46 = iaa.Grayscale(alpha=(0.0, 0.2))
# # aug47 = iaa.ChangeColorTemperature((1100, 10000))
# # aug49 = iaa.UniformColorQuantization()
# # aug50 = iaa.UniformColorQuantizationToNBits()
# aug51 = iaa.GammaContrast((0.1, .5), per_channel=True)
# aug52 = iaa.SigmoidContrast(gain=(1, 3), cutoff=(0.1, 0.3), per_channel=True)
# aug53 = iaa.LogContrast(gain=(0.6, .9), per_channel=True)
# aug54 = iaa.LinearContrast((0.4, .9), per_channel=True)
# # aug55 = iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True)
# aug56 = iaa.Alpha((0.0, 1.0), iaa.AllChannelsHistogramEqualization())
# # aug57 = iaa.HistogramEqualization(
# #     from_colorspace=iaa.HistogramEqualization.BGR,
# #     to_colorspace=iaa.HistogramEqualization.HSV)

# # aug58  = iaa.DirectedEdgeDetect(alpha=(0.0, 0.5), direction=(0.0, 0.5))
# # aug59 = iaa.Canny(
# #     alpha=(0.0, 0.1),
# #     colorizer=iaa.RandomColorsBinaryImageColorizer(
# #         color_true=255,
# #         color_false=0
# #     )
# # )

scale=(0, 20)
aug1 = iaa.Add((-20, 20))
aug2 = iaa.AddElementwise((-20, 20), per_channel=0.5)
aug3 = aug = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))

aug4 = iaa.AdditiveLaplaceNoise(scale=(0, 0.2*255))
aug5 = iaa.AdditivePoissonNoise(scale)
aug6 = iaa.Multiply((0.5, 1.5), per_channel=0.5)
aug7 = iaa.Cutout(nb_iterations=2, size=0.05)

aug8 = iaa.Cutout(fill_mode="constant", size=0.05, cval=255)

aug9 = iaa.Cutout(fill_mode="gaussian", fill_per_channel=True, size=0.05)
aug10 = iaa.Dropout(p=(0, 0.05))
aug11 = iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.2))
aug12 = iaa.Dropout2d(p=0.05, nb_keep_channels=0)
aug13 = iaa.ImpulseNoise(0.1)
aug14 = iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.6))
aug15 = iaa.CoarsePepper(0.05, size_percent=(0.01, 0.2))
# aug16 = iaa.Invert(0.25, per_channel=0.4)
# aug17 = iaa.Invert(0.1)
# aug18 = iaa.Solarize(0.05, threshold=(32, 128))
# aug19 = iaa.JpegCompression(compression=(95, 99))
aug20 = iaa.GaussianBlur(sigma=(0.0, 3.0)) # blur
aug21 = iaa.AverageBlur(k=((1, 5), (1, 3)))

aug22 = iaa.MotionBlur(k=3)

# aug23 = iaa.BlendAlpha(
#     (0.0, 0.5),
#     iaa.Affine(rotate=(0, 0)),
#     per_channel=0.5)
# aug24 = iaa.BlendAlpha(
#     (0.0, 0.5),
#     foreground=iaa.Add(60),
#     background=iaa.Multiply(0.2))

aug25 = iaa.BlendAlphaMask(
    iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()),
    iaa.Clouds()
)

aug26 = iaa.BlendAlphaElementwise(
    (0.0, 0.5),
    iaa.Affine(rotate=(-0, 0)),
    per_channel=0.5)

aug27 = iaa.BlendAlphaElementwise(
    (0.0, 0.2),
    foreground=iaa.Add(20),
    background=iaa.Multiply(0.4))
aug28 = iaa.BlendAlphaElementwise([0.25, 0.75], iaa.MedianBlur(13))

aug29 = iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(0.6))
aug30 = iaa.BlendAlphaSimplexNoise(
    iaa.EdgeDetect(0.2),
    upscale_method="nearest")
aug31 = iaa.Cutout(fill_mode="constant", cval=(0, 255),
                  fill_per_channel=0.5)
# aug32 = iaa.BlendAlphaHorizontalLinearGradient(
#     iaa.TotalDropout(0.3),
#     min_value=0.1, max_value=0.3)

# aug33 = iaa.BlendAlphaHorizontalLinearGradient(
#     iaa.AveragePooling(3),
#     start_at=(0.0, 0.2), end_at=(0.0, 0.2))

# aug34 = iaa.BlendAlphaVerticalLinearGradient(
#     iaa.TotalDropout(0.6),
#     min_value=0.2, max_value=0.8)

# aug35 = iaa.BlendAlphaVerticalLinearGradient(
#     iaa.AveragePooling(9),
#     start_at=(0.0, 0.4), end_at=(0.0, 0.4))

# aug36 = iaa.BlendAlpha(
#     (0.0, 0.3),
#     iaa.Affine(rotate=(0, 0)),
#     per_channel=0.3)

aug38 =  iaa.WithColorspace(
    to_colorspace="HSV",
    from_colorspace="RGB",
    children=iaa.WithChannels(0,iaa.Add((0, 50))))
aug40 = iaa.WithHueAndSaturation(
    iaa.WithChannels(0, iaa.Add((0, 50))))
aug41 = iaa.MultiplyHueAndSaturation((0.5, 1.9), per_channel=True)
aug42 = iaa.AddToHueAndSaturation((-50, 50), per_channel=True)
aug43 = iaa.AddToHue((-50, 50))
aug44 = iaa.AddToSaturation((-50, 50))
aug45 = iaa.Sequential([
    iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
    iaa.WithChannels(0, iaa.Add((50, 100))),
    iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")])
    
aug46 = iaa.Grayscale(alpha=(0.0, 1.0))
aug47 = iaa.ChangeColorTemperature((1100, 10000))
aug49 = iaa.UniformColorQuantization()
aug50 = iaa.UniformColorQuantizationToNBits()
aug51 = iaa.GammaContrast((0.5, 2.0), per_channel=True)
aug52 = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True)
aug53 = iaa.LogContrast(gain=(0.6, 1.4), per_channel=True)
aug54 = iaa.LinearContrast((0.4, 1.6), per_channel=True)
# aug55 = iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True)
aug56 = iaa.Alpha((0.0, 1.0), iaa.AllChannelsHistogramEqualization())
aug57 = iaa.HistogramEqualization(
    from_colorspace=iaa.HistogramEqualization.BGR,
    to_colorspace=iaa.HistogramEqualization.HSV)

aug58  = iaa.DirectedEdgeDetect(alpha=(0.0, 0.5), direction=(0.0, 0.5))
aug59 = iaa.Canny(
    alpha=(0.0, 0.3),
    colorizer=iaa.RandomColorsBinaryImageColorizer(
        color_true=255,
        color_false=0
    )
)

def aug_imgaug(aug, image):

    image2 = image.copy()
    image2 = np.expand_dims(image2, axis=0)
    images_aug = aug(images = image2)
    
    return images_aug

class FaceEmbeddings():
  """Class to load  model and run inference."""

  INPUT_TENSOR_NAME = 'input_image:0'
  OUTPUT_TENSOR_NAME = 'embedding_layer/MatMul:0'
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self,model_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    
    file_handle = open(model_path,  "rb")
    graph_def = tf.GraphDef.FromString(file_handle.read())

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    
    embeddings = self.sess.run(self.OUTPUT_TENSOR_NAME,feed_dict={self.INPUT_TENSOR_NAME: [image]})
    
    return embeddings

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

fa = FaceAligner(desiredFaceWidth=150)
    
class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)


def normalize_image(image):    

    [B,G,R] = np.dsplit(image, image.shape[-1])

    Rx = (R - 122.782) / 256.
    Gx = (G - 117.001) / 256.
    Bx = (B - 104.298) / 256.

    new_image = np.dstack((Rx, Gx, Bx))
    return new_image

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths    

def get_dataset(path, has_class_directories = True):
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
    images = {}
    labels=[]
    for i in range(len(dataset)):
        image_paths = dataset[i].image_paths
        key = i
        li = []
        labels.append(i)
        for image_path in image_paths:

            org_image2 = cv2.imread(image_path)

            try:
                result = detector.detect_faces(org_image2)
            except:
                print('Something wrong with MTCNN')
                continue
            
            if len(result) == 0:
                continue

            for res in result:
                x_min, y_min, x_max, y_max = res['box']
            
                keypoints = res['keypoints']    
                faceAligned = fa.align(org_image2, keypoints)        
                
                if faceAligned is None:
                    continue
                
                # norm_img = normalize_image(faceAligned.copy())
                img1 = cv2.resize(faceAligned, (150, 150))
                img2 = img1[...,::-1]
                li.append(img2)
            # li.append(np.transpose(img2, (2,0,1)))
        images[key] = np.array(li)    
        print(key)
    return images, labels

dataset = get_dataset('train_celeb/')
images, labels = get_image_paths_and_labels(dataset)
my_list = [aug1, aug2, aug3 ,aug4 ,aug5 ,aug6 ,aug7 ,aug8 ,aug9 ,aug10 ,aug11 ,aug12 ,aug13 , aug14, aug15
           ,aug22 ,aug25 ,aug26 ,aug27 ,aug28 ,aug29 ,aug30 ,aug31 
            ,aug38 ,aug40 ,aug41 ,aug42 ,aug43 ,aug44  , aug46 ,           
           aug20, aug21, aug31, aug46, aug51, aug52, aug53, aug54, aug56]

# images = np.load("images.pkl")
# labels = np.load("labels.pkl")

faces = labels
face_embeddings = FaceEmbeddings('model/dlib_face_recognition_resnet_model_v1.pb')

IMAGE_SIZE = 150
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

def batch_generator(batch_size=16):

    y_val = np.zeros((batch_size, 2, 1))
    anchors = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    positives = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    negatives = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    
    while True:

        for i in range(batch_size):

            positiveFace = faces[np.random.randint(len(faces))]
            negativeFace = faces[np.random.randint(len(faces))]
            
            while positiveFace == negativeFace:
                negativeFace = faces[np.random.randint(len(faces))]

            dist_a_p = .1
            dist_a_n = .4
            
            while(dist_a_p - dist_a_n <= -.2):
            
                x1 = np.random.randint(len(images[positiveFace]))
                x2 = np.random.randint(len(images[positiveFace]))
                # print('x1 ', x1)
                # print('x2 ', x2)

                positive = images[positiveFace][x1]
                anchor = images[positiveFace][x2]
                negative = images[negativeFace][np.random.randint(len(images[negativeFace]))]
                
                pos_aug_func = random.choice(my_list)
                # print('pos_aug_func >> ', pos_aug_func)

                anc_aug_func = random.choice(my_list)
                # print('anc_aug_func >> ', anc_aug_func)

                neg_aug_func = random.choice(my_list)
                # print('neg_aug_func >> ', neg_aug_func)
                
                pos_img = cv2.cvtColor(aug_imgaug(pos_aug_func, positive)[0, :, :, :], cv2.COLOR_BGR2RGB)
                positive_aug = normalize_image(pos_img)
                
                anc_img = cv2.cvtColor(aug_imgaug(anc_aug_func, anchor)[0, :, :, :], cv2.COLOR_BGR2RGB)
                anchor_aug = normalize_image(anc_img)
                
                negative_img = cv2.cvtColor(aug_imgaug(neg_aug_func, negative)[0, :, :, :], cv2.COLOR_BGR2RGB)
                negative_aug = normalize_image(negative_img)
                
                positive_embeddings = face_embeddings.run(positive_aug)
                anchor_embeddings = face_embeddings.run(anchor_aug)
                negative_embeddings = face_embeddings.run(negative_aug)
    
                dist_a_p = np.linalg.norm(anchor_embeddings - positive_embeddings)
                dist_a_n = np.linalg.norm(anchor_embeddings - negative_embeddings)
                # pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor_embeddings, positive_embeddings)), axis=-1)
                
                # print('dist_a_p ', dist_a_p)
                # print('pos_dist ', pos_dist)
                
                # print(np.linalg.norm(dist_a_p-dist_a_n))
            
            print('Finall ' , dist_a_p-dist_a_n)
                            
            # cv2.imshow('str(pos_aug_func)', pos_img)
            # cv2.imshow("str(anc_aug_func)", anc_img)
            # cv2.imshow("str(neg_aug_func)", negative_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            anchors[i] = anchor_aug
            positives[i] = positive_aug
            negatives[i] = negative_aug

        x_data = { 'anchor': anchors,
                   'anchorPositive': positives,
                   'anchorNegative': negatives
                 }
        
        # print(x_data)
        yield (x_data, [y_val, y_val, y_val])
        
# batch_generator(batch_size=4)   

# x = x_data['anchor'][3]
# y = np.transpose(x, (2,1,0))
# cv2.imshow('as', y.astype(np.int8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()