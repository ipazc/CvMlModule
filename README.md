# CVMLMODULE 
Computer Vision and Machine Learning API-REST module.

This module intends to serve as an API-REST cloud-based platform for applying CVML algorithms to resources on the fly and retrieve a result.

# INSTALLATION

It is required Ubuntu 14.04 at least, with Caffe, Opencv3 and DLib installed. It is required Flask in Python3 to be installed in order to run the framework.
CNN Caffe models have been removed due to its size. This implies that face detection is only available with DLib and OpenCV algorithms.
Age and gender estimation are also not available due to the removed models. Contact me at ipazc@unileon.es if you want to test the missing ones.

# HOW TO RUN

Go to the root folder and execute the entry file located at main/bin:
```bash
python3 -m main.bin.entry
```

# EXAMPLES

The following examples explains how to use the API-Rest from CURL calls. Replace 192.168.2.110:9095 with your IP and port deployed by `entry.py`.

## Detect faces

### Show available algorithms

```bash
curl 'http://192.168.2.110:9095/detection-requests/faces/services' -s -X GET | jq '.'
```

### Get BBoxes of faces
```bash
curl 'http://192.168.2.110:9095/detection-requests/faces/stream' -s -X PUT --data-binary @"uri-to-file.jpg" | jq '.'
```

### Get BBoxes of faces with other algorithms
```bash
curl 'http://192.168.2.110:9095/detection-requests/faces/stream?service=SERVICE_NAME' -s -X PUT --data-binary @"uri-to-file.jpg" | jq '.'
```


## Estimate ages

### Show available algorithms

```bash
curl 'http://192.168.2.110:9095/estimation-requests/age/services' -s -X GET | jq '.'
```

### Get Age range of face
```bash
curl 'http://192.168.2.110:9095/estimation-requests/age/face/stream' -s -X PUT --data-binary @"uri-to-file.jpg" | jq '.'
```

### GetAge range of face with other algorithms
```bash
curl 'http://192.168.2.110:9095/estimation-requests/age/face/stream?service=SERVICE_NAME' -s -X PUT --data-binary @"uri-to-file.jpg" | jq '.'
```

## Estimate genders

### Show available algorithms

```bash
curl 'http://192.168.2.110:9095/estimation-requests/gender/services' -s -X GET | jq '.'
```

### Get gender of face
```bash
curl 'http://192.168.2.110:9095/estimation-requests/gender/face/stream' -s -X PUT --data-binary @"uri-to-file.jpg" | jq '.'
```

### Get gender of face with other algorithms
```bash
curl 'http://192.168.2.110:9095/estimation-requests/gender/face/stream?service=SERVICE_NAME' -s -X PUT --data-binary @"uri-to-file.jpg" | jq '.'
```


## Detect pedestrians

### Show available algorithms

```bash
curl 'http://192.168.2.110:9095/detection-requests/pedestrians/services' -s -X GET | jq '.'
```

### Get BBoxes of pedestrians
```bash
curl 'http://192.168.2.110:9095/detection-requests/pedestrians/stream' -s -X PUT --data-binary @"uri-to-file.jpg" | jq '.'
```

### Get BBoxes of pedestrians with other algorithms
```bash
curl 'http://192.168.2.110:9095/detection-requests/pedestrians/stream?service=SERVICE_NAME' -s -X PUT --data-binary @"uri-to-file.jpg" | jq '.'
```


## Ensemble face-age-gender recognition

### Show available algorithms

```bash
curl 'http://192.168.2.110:9095/ensemble-requests/faces/services' -s -X GET | jq '.'
```

### Get BBoxes of faces + ageranges + genders
```bash
curl 'http://192.168.2.110:9095/ensemble-requests/faces/detection-estimation-age-gender/stream' -s -X PUT --data-binary @"uri-to-file.jpg" | jq '.'
```

### Get BBoxes of faces +ageranges + genders with other algorithms
```bash
curl 'http://192.168.2.110:9095/ensemble-requests/faces/detection-estimation-age-gender/stream?service_face=SERVICE_NAMEF&service_age=SERVICE_NAMEA&service_gender=SERVICE_NAMEG' -s -X PUT --data-binary @"uri-to-file.jpg" | jq '.'
```

## Draw bboxes
### Draw bboxes onto an image.
```bash
curl 'http://192.168.2.110:9095/draw-requests/boundingboxes/stream?bounding_boxes=(color_line-height)X,Y,Width,Height;(color_line-height)X,Y,Width,Height...' -s -X PUT --data-binary @"uri-to-file.jpg" > output_file.jpg
```
