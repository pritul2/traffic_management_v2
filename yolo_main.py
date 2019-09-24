import cv2
import numpy as np



def detect(img):
	#passing yolo weights and cfg#
	NN=cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")
	layer_name=NN.getLayerNames()

	output_layer = [layer_name[i[0] - 1] for i in NN.getUnconnectedOutLayers()]

	#reading the dataset#
	file=open("coco.names")
	classes=[]
	for ch in file.readlines():
		classes.append(ch.strip('\n'))

	#img=cv2.resize(img,(416,416),interpolation=cv2.INTER_AREA)

	#detecting the objects present in image#
	blobs = cv2.dnn.blobFromImage(img,1 / 255.0, (416, 416),swapRB=True, crop=False)
	#passing the detected blobs to neural network#
	NN.setInput(blobs)
	#getting output#
	output_neuron=NN.forward(output_layer)
	(H, W) = img.shape[:2]
	boxes = []
	confidences = []
	classIDs = []
	for out in output_neuron:
		for detection in out:
			scores=detection[5:]
			class_id=np.argmax(scores)
			confidence=scores[class_id]
			if confidence>=0.2 and (classes[class_id] in ['car','bus','motorbike','truck']):
				#print(detection[1],img.shape[1])
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(class_id)

	#Remove the overlapped rectangle through threshold#
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2,0.3)
	count=0
	for i in range(len(boxes)):
		if i in indexes.flatten():
			count=count+1

	return count
