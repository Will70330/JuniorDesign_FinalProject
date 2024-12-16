# JuniorDesign_FinalProject

## Using Fusion 360 for the Design

### The Base

The base of the design is meant to house a Raspberry Pi alongside the actual servo that will be rotating the head. The Raspberry Pi sits within the enclosure with added holes for both ventilation and various I/O Ports (ie. HDMI, USB, power, etc) as can be seen in the images below.

![Base Top View](./examples/Base_TopView.jpg)
![Base Angled View](./examples/Base_AngleView.jpg)
![Lid Angled View](./examples/Lid_AngleView.jpg)

This is what essentially acts as our base for the head to rest on while sitting atop someone's desk.

### The Head

The head has been designed to fit up to a RealSense d455 Depth Camera sensor. There is an extruded platform on which the sensor rests, with enough space to be able to actually plug in the camera from below still. Additionally, there is a hole exiting the back of the head for which the connecting cables can exit and plug into the Raspberry Pi housed in the base.

Additional holes are placed in the front fascia where there is supposed to be an integrated speaker, but for now, the speaker was not added, so it serves to act as some additonal ventilation for the camera sensor.

Unfortunately, my CADing skills are still very poor, so the design of the head itself is very limited. However, it did result in a very cool Iron Man Prototype-esque appearance that looks awesome to some, and probably terrifying to most.

![Outside of Frontal Head Piece](./examples/Head_Front_Outside.jpg)
![Inside of Frontal Head Piece](./examples/Head_Front_Inside.jpg)
![Inside of Back Head Piece](./examples/Head_Back_Inside.jpg)

## Using OpenCV for Facial Detection

### [Haar Cascades](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)

Object Detection using Haar feature-based cascade classifiers was an effective object detection method proposed by Paul Viola & Micahel Joones in their paper: ["Rapid Object Detection using a Boosted Cascade of Simple Features"](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf) in 2001.

This machine learning approach uses a cascade function trained from numerous images with and without faces. For the feature extraction, Haar features are used which can be seen below.

![Haar Features](./examples/haar_features.jpg)

These kernels are used to compute the features of a given frame by summing the pixels under the white and black rectangles, but even for small frames (24x24), this generates an enormous amount of features (160,000). This was solved with the integral image where the calculation for each pixel was solved with only 4 pixels, making things faster.

As the model is trained, the features that are most accurate in classifying faces are weighed highly while the rest are not. The final classifier is a weighted sum of the weak classifiers that normally cannot classify an image on their own. Overall, the paper showed great results with using just 200 features (95% accuracy), but ended up with ~6000 features in the final setup.

However, there was still a huge inefficiency, for a given image, you apply all 6000 features to it. This is inefficient because most of the image is not going to have a face, so we waste a lot of compute searching over the entire image. Instead, we can check whether or not a window within the image is likely to be a face region, if it's not, discard it in a single shot and don't process it again.

This is where the **Cascade of Classifiers** comes into play, instead of applying all 6000 features on a window, the features are grouped into stages of classifiers that are applied one-by-one. If a window fails in the first stage, throw it out so we don't consider the remaining features. The paper used 6000+ features spread across 38 different stages (1, 10, 25, 25, and 50 in the first 5 stages).

Luckily, OpenCV provides methods for training your own Cascade Classifier model or using a pretrained model. For the purposes of this project, I investigated using a pretrained model. The results of which can be found below.

![Haar Cascade Facial Detection Model](./examples/FacialDetection_Cascades.gif)

As we can see, the model has significant false positives and also fails to detect non-frontal faces. There are multiple reasons for this, the main one being that we are using the frontal face pretrained model. If we trained our own model with our own dataset, we could likely achieve better non-frontal detection performance, but that is out of the scope of this project.

### [YuNet](https://link.springer.com/article/10.1007/s11633-023-1423-y)

This is where [YuNet](https://link.springer.com/article/10.1007/s11633-023-1423-y) comes in. The YuNet model came from Wei Wu, Hanyang Peng, and Shiqi Yu who recognized a need for fast and accurate Facial Detection models. They developed a lightweight facial detection model that was designed for mobile and embedded device applications that had limited compute resources, perfect for this project.

YuNet achieved a strong balance between accuracy and speed at the millisecond-level while significantly reducing the parameters (75,856) and computational costs, making it a fifth the size of other small face detectors.

The architecture uses depthwise separable convolution with Tiny Feature Pyramid Network for combining multiscale features. Their detection head is an anchor-free mechanism that simplifies predictions by reducing candidate locations, enabling faster inference and lower computation requirements. As a result, YuNet saw a mAP of 81.1% on the WIDER FACE validation hard track while maintaining a millisecond-level inference speed, outperforming other models in terms of speed and efficiency.

Luckily, OpenCV also has a pretrained YuNet model readily available for use, resulting in a much more accurate and stable result that works with minor occlusions and non-frontal facial positions as seen below!

![YuNet Facial Detection Model](./examples/FacialDetection_YuNet.gif)

## Using [Gemini](https://ai.google.dev/gemini-api/docs/quickstart?_gl=1*17d54za*_up*MQ..&gclid=CjwKCAiAmfq6BhAsEiwAX1jsZ0pijycy7uQXAYtBiWm_CS0-SJHGn6CynoKkWXzQRwCfrn1JO_HbJRoCefsQAvD_BwE&lang=python) for Facial Recognition and Temporal Tracking

In the competitive space of AI and Large Language Models (LLMs), there are many different competitors fighting for marketshare. Some of the biggest names in the space at this moment are OpenAI's ChatGPT, Anthropic's Claude, Meta's Llama, and Google's Gemini. Of these models, I've elected to utilize Google's Gemini API with both the Flash (free and light weight) and Pro (paid, larger, more advanced) models from their Gemini 1.5 lineup. One of the primary benefits of Gemini is it's 2M token context window that allows for long-context conversational chats, something that this application should benefit from. As an AI desk assistant, we don't need to remember ALL of the chat history, but if we are tokenizing image frames for input into the model, we will want to keep track of users and the context of their surroundings and interactions. Long-Context should enable us to have longer chat sessions without losing memory because each image is scaled to fit within Gemini's resolution limits (minimum of 768 pixels, maximum of 3072) and capped at 258 tokens.

The Gemini API is very well documented and their flash and pro models are some of the only models to offer multi-modal capabilities that span a significant number of modalities (ie. text, images, videos, audio). My time at Google allowed me to learn more about the API and its limitations and use cases. By using the Gemini API, I don't need to worry about quanitizing any largescale models to fit within limited computational resources. By calling the API directly, I can leverage Google's datacenters and compute in order to process queries and produce tokens faster than I could locally on a laptop or raspberry pi.

However, there are still downsides to this approach. The first being data security, with this solution, image frames and query data are all likely shared with Google as we are using their APIs. Additionally, there is significantly more latency as we have to send the query to their servers and then wait for the tokens to be produced and sent back to our laptop, therefore this prevents our system from running at ultra low latency and polling frames quickly because of the network overhead. Although, this is still significantly faster than running any larger models locally as output tokens would not be produced nearly as quickly.

So we've discussed using Cascades and YuNet for Facial Detection, but what about actually identifying the person in frame? For that, we need some way to preserve termporal context if we aren't going to use tracking. We can accomplish this with Gemini through custom structure response outputs and preemptively loading the model with identities. For example, if I provide the model with the following images.

![Temporal Test Image 1](./examples/temporal_test1.png) ![Temporal Test Image 1](./examples/temporal_test2.png)

I can tell the model that this is Will. Theoretically, it should then be capable of detecting and knowing Will better as the conversation continues. Furthermore, this can extend to objects as well, where we can prompt Gemini about objects that are also within the frame.

For our given test case, we specify a structured type "Entity" as defined below.

```python
class Entity(typing.TypedDict):
    entity_name: str
    entity_type: str
    entity_location: list[int]
```

We can then specify the return type for the model with a configuration as such:

```python
# Set up Model API
genai.configure(api_key=gemini_api_key)
conf = genai.types.GenerationConfig(
    temperature=1.0,
    response_mime_type="application/json",
    response_schema=list[Entity]   
)
identifier_model = genai.GenerativeModel("models/gemini-1.5-flash", generation_config=conf)
```

Below is an example chat with Gemini providing the last 2 images as inputs with the following prompts. We first start with an initial prompt to tell Gemini the behavior we want for this particular "agent".

```python
# Start Interaction
chat = identifier_model.start_chat(history=[])
chat.send_message("""
    You are an identifier, your job is to identify people and objects within images.
    Provide answers that are concise and accurate, and do not hallucinate or make up any answers.
    
    For the entity name:
        if the entity is a person, fill with some random identifier ID unless provided with a specific name.
        if the entity is an object, fill with the name of the object (ie phone, pen, apple, etc).
                    Try to be specific as possible when identifying objects (iPhone vs. Phone, Red Shirt vs Shirt, Black Marker vs Pen)     
    For the entity location, use a list in [ymin, xmin, ymax, xmax] format.
    For the entity type, use 'person' for people, and 'item' for objects.
                    
    List all relevant entities (people and objects) within the frame that are relevant to the query.
""")
```

Here is the following chat after the agent has been primed:

```
________________________________________________________________________________
Prompt: input_image1 + input_image2 + "The person in this image is William."
Model Response:
 [{"entity_location": [186,486,500,637], "entity_name": "William", "entity_type": "person"}, {"entity_location": [0,0,0,0], "entity_name": "Over-ear Headphones", "entity_type": "item"}, {"entity_location": [0,0,0,0], "entity_name": "Red Nike T-Shirt", "entity_type": "item"}, {"entity_location": [0,0,0,0], "entity_name": "Eyeglasses", "entity_type": "item"}]
________________________________________________________________________________
Prompt: input_image3 + "Who is holding the phone?"
Model Response:
 [{"entity_location": [166,406,603,682], "entity_name": "William", "entity_type": "person"}, {"entity_location": [327,678,625,851], "entity_name": "iPhone 12 with MagSafe Case", "entity_type": "item"}]
________________________________________________________________________________
Prompt: "What type of phone is he holding"
Model Response:
 [{"entity_location": [327,678,625,851], "entity_name": "iPhone 12 with MagSafe Case", "entity_type": "item"}]
```

We can see with just the first prompt, the model was able to detect various objects that were on or around William (me) such as "Over-ear headphones", "Red Nike T-Shirt", and "Glasses".

By telling Gemini that we also want a bounding box for the answer of each entity, we can then perform a simple calculation to actually convert this bounding box to the coordinate frame of our original image. First we just divide each coordinate by 1000, and then we multiply the x coordinates by the width of the original image and we multiple the y coordinates by the height of the original image. Using these coordinates for the bounding box, we can draw a rectangle representing where Gemini thinks the object is as well as labeling it with the name Gemini thinks it is as seen below:

![Gemini Bounding Box Output](./examples/ObjectDetection_Gemini.jpg)

The detection and predictions are not perfect or consistent, but can likely be dialed in with tuning the temperature value for the model responses, but this is out of scope for the timeframe of this project.

## References
- https://www.intelrealsense.com/sdk-2/
- https://github.com/IntelRealSense/librealsense/releases/tag/v2.56.3
- https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
- https://www.youtube.com/watch?v=CmDO-w56qso&ab_channel=EngineeringCorner
- https://www.datacamp.com/tutorial/face-detection-python-opencv
- https://ai.google.dev/gemini-api/docs?gad_source=1&gclid=CjwKCAiAmfq6BhAsEiwAX1jsZ0pijycy7uQXAYtBiWm_CS0-SJHGn6CynoKkWXzQRwCfrn1JO_HbJRoCefsQAvD_BwE
- https://link.springer.com/article/10.1007/s11633-023-1423-y
- https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf
- https://www.kaggle.com/code/junhyeonkwon/using-yunet
- https://discuss.ai.google.dev/t/how-to-make-a-conversation-with-gemini-that-supports-pictures/52472
- 
