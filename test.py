import json
#from os.path import join, dirname
#from os import environ
from watson_developer_cloud import VisualRecognitionV3
import cv2
import numpy as np
import urllib
 
# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image


#test_url = 'https://www.ibm.com/ibm/ginni/images/ginni_bio_780x981_v4_03162016.jpg'
test_url = 'http://indiarainbow.org/wp-content/uploads/2015/02/dreamstime_m_337094551.jpg'

visual_recognition = VisualRecognitionV3('2016-05-20', api_key='71a5ce05e2c676ac2a42b09751fd474393b5c95c')

# with open(join(dirname(__file__), '../resources/cars.zip'), 'rb') as cars, \
#        open(join(dirname(__file__), '../resources/trucks.zip'), 'rb') as trucks:
#     print(json.dumps(visual_recognition.create_classifier('Cars vs Trucks', cars_positive_examples=cars,
#                                                           negative_examples=trucks), indent=2))

# with open(join(dirname(__file__), 'resources/car.jpg'), 'rb') as image_file:
#     print(json.dumps(visual_recognition.classify(images_file=image_file, threshold=0.1,
#                                                  classifier_ids=['CarsvsTrucks_1675727418', 'default']), indent=2))

# print(json.dumps(visual_recognition.get_classifier('YOUR CLASSIFIER ID'), indent=2))

# with open(join(dirname(__file__), '../resources/car.jpg'), 'rb') as image_file:
#     print(json.dumps(visual_recognition.update_classifier('CarsvsTrucks_1479118188',
#                                                           cars_positive_examples=image_file), indent=2))

res = visual_recognition.classify(images_url=test_url)
j1 = eval(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
print 'Image is about:', j1['images'][0]['classifiers'][0]['classes'][0]['class']

res2 = visual_recognition.detect_faces(images_url=test_url)
j2 = eval(json.dumps(res2, indent=2))
print(json.dumps(res2, indent=2))
print 'Number of faces:', len(j2['images'][0]['faces'])
print 'Gender:', j2['images'][0]['faces'][0]['gender']['gender']
print 'Age:', j2['images'][0]['faces'][0]['age']
print 'Face location:', j2['images'][0]['faces'][0]['face_location']


img = url_to_image(test_url)
cv2.imshow('Original', img)

for idx in range(len(j2['images'][0]['faces'])):
    x = j2['images'][0]['faces'][idx]['face_location']['left']
    y = j2['images'][0]['faces'][idx]['face_location']['top']
    w = j2['images'][0]['faces'][idx]['face_location']['width']
    h = j2['images'][0]['faces'][idx]['face_location']['height']
    gender = j2['images'][0]['faces'][idx]['gender']['gender']
    age = '(' + str(j2['images'][0]['faces'][idx]['age']['min']) + '-' + str(j2['images'][0]['faces'][idx]['age']['max']) + ')'
    if (gender == 'MALE'):
        cv2.rectangle(img, (x,y), (x+h, y+w), (255,0,0), 2)
        cv2.putText(img, gender+age, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),2,cv2.CV_AA)
    else:
        cv2.rectangle(img, (x,y), (x+h, y+w), (0,0,255), 2)
        cv2.putText(img, gender+age, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),2,cv2.CV_AA)

cv2.imshow('Face', img)


cv2.waitKey(0)
cv2.destroyAllWindows()



# print(json.dumps(visual_recognition.delete_classifier(classifier_id='YOUR CLASSIFIER ID'), indent=2))

#j3 = json.dumps(visual_recognition.list_classifiers(), indent=2)
#print j3

# with open(join(dirname(__file__), 'resources/text.png'), 'rb') as image_file:
#     print(json.dumps(visual_recognition.recognize_text(images_file=image_file), indent=2))
#
# with open(join(dirname(__file__), 'resources/face.jpg'), 'rb') as image_file:
#     print(json.dumps(visual_recognition.detect_faces(images_file=image_file), indent=2))
