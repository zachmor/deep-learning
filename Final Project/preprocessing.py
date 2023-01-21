import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET

# if runned
def get_data(datagen=False, bs=0):
    # Instantiating list of images, one hots
    images = []
    one_hots = []
    

    # Reading images, annotations in each class directory
    annotation_class_paths = os.listdir("Annotation")
    images_class_paths = os.listdir("Images")
    zipped = zip(annotation_class_paths, images_class_paths)
    for i, (images_class_path, annotation_class_path) in enumerate(zipped):
        images_class_path = "Images/" + images_class_path
        annotation_class_path = "Annotation/" + annotation_class_path
        image_paths = os.listdir(images_class_path)
        annotation_paths = os.listdir(annotation_class_path)
        zipped = zip(image_paths[:10], annotation_paths[:10])
        for image_path, annotation_path in zipped:
            image_path = images_class_path + "/" + image_path
            annotation_path = annotation_class_path + "/" + annotation_path
            # Reading image
            image = cv2.imread(image_path)
            # Getting bounding box coordinates
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            x_min = int(root.find('object').find('bndbox').find('xmin').text)
            y_min = int(root.find('object').find('bndbox').find('ymin').text)
            x_max = int(root.find('object').find('bndbox').find('xmax').text)
            y_max = int(root.find('object').find('bndbox').find('ymax').text)
            # Cropping image to bounding box
            image = image[x_min:x_max, y_min:y_max]
            # Omitting images where either valid region dim < 256
            if image.shape[0] > 255 and image.shape[1] > 255:
                # Cropping to top left 256x256 square 
                images.append(image[:256,:256,:])
                # Adding corresponding one hot vector
                one_hot = np.zeros(120)
                one_hot[i] = 1
                one_hots.append(one_hot)

    images = np.array(images).astype(np.float32)/255
    one_hots = np.array(one_hots).astype(np.float32)
    images_per_class = map(len, images)

    print("shape of one hots: ", one_hots.shape)
    print("Shape of imaegs: ", images.shape)
    print("image shape: ", images[0].shape)
    '''
    for c in class_images:
        print(c.shape)
    '''
    num_inputs = len(images)
    num_train = int(round(num_inputs * 0.6))
    train_inputs, test_inputs = images[:num_train], images[num_train:]
    train_labels, test_labels = one_hots[:num_train], one_hots[num_train:]



# Input an image, return a randomly augmented one,
# with l/r flip, contrast, brightness, jittering, 
    if datagen == True: #return datagen object instead of training stuff
        dg = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, zca_whitening=True, rotation_range=20, width_shift_range=.05, height_shift_range=.05, horizontal_flip=True)
        dg.fit(train_inputs)
        return dg.flow(train_inputs, train_labels, batch_size=bs), test_inputs, test_labels

    return train_inputs, train_labels, test_inputs, test_labels
