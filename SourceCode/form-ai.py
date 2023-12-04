#IMPORTS
import tensorflow as tf
import cv2
from keras.preprocessing.image import ImageDataGenerator
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.uix.label import Label
from posenet import PoseNet
from kivy.graphics.texture import Texture 
import numpy as np

#VARIABLES
img_dataset_path='C:\Users\shrey\Desktop\Web_projects\miniProject\FormAI\dataset'
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
vid_dataset_path='C:\Users\shrey\Desktop\Web_projects\miniProject\FormAI\dataset'
frames_save_path='C:\Users\shrey\Desktop\Web_projects\miniProject\FormAI\Frames'
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
batch_size = 50
image_height = 128
image_width = 128
net=PoseNet()
result='Correct Form'
model = tf.keras.models.load_model('workout_assessment_model.h5')

#DATASET LOADING AND PREPROCESSING
train_generator = datagen.flow_from_directory(
    img_dataset_path,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical', 
    subset='training'
)
validation_generator = datagen.flow_from_directory(
    img_dataset_path,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
def load_and_preprocess_video(video_path):
    
    video_frames = tf.io.read_file(video_path)
    video_frames = tf.image.decode_video(video_frames)
    
    frames = video_frames['video']
    
    frames = tf.image.convert_image_dtype(frames, dtype=tf.float32)    
    return frames

video_dataset = tf.keras.utils.image_dataset_from_directory(
    vid_dataset_path,
    labels='inferred',
    batch_size=batch_size,
    image_size=(image_height, image_width),
    validation_split=0.2,
    subset='training',
    seed=123,
    label_mode='categorical'
)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')  # Assuming two classes: correct form and incorrect form
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=5,  # You can adjust the number of epochs
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)
model.save('workout_assessment_model.h5')

#FUNCTIONS TO CAPTURE AND ANALYSE USER INPUT AND PRODUCE OUTPUT

def preprocessed_keypoints(keypoints):
    shoulder = keypoints.get('shoulder', [0, 0])
    elbow = keypoints.get('elbow', [0, 0])
    wrist = keypoints.get('wrist', [0, 0])
    hip = keypoints.get('hip', [0, 0])
    knee = keypoints.get('knee', [0, 0])
    ankle = keypoints.get('ankle', [0, 0])

    shoulder = np.array(shoulder)
    elbow = np.array(elbow)
    wrist = np.array(wrist)
    hip = np.array(hip)
    knee = np.array(knee)
    ankle = np.array(ankle)
    
    return shoulder, elbow, wrist, hip, knee, ankle

def capture_video():
    global result
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't open the camera.")
        return
    while True:
        ret, frame=cap.read()
        keypoints, _=net(frame)
        if keypoints is not None:
            frame=PoseNet.draw_keypoints(frame, keypoints)
            result=assess_workout_form(keypoints)
            cv2.putText(frame, result, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def assess_workout_form(keypoints):
    nose_index=0
    left_shoulder_index=5
    right_shoulder_index=6
    left_hip_index=11
    right_hip_index=12
    nose=keypoints[nose_index]
    left_shoulder=keypoints[left_shoulder_index]
    right_shoulder=keypoints[right_shoulder_index]
    left_hip=keypoints[left_hip_index]
    right_hip=keypoints[right_hip_index]
    shoulders_aligned=abs(left_shoulder[1]-right_shoulder[1])<20
    hips_aligned=abs(left_hip[1]-right_hip[1])<20
    if shoulders_aligned and hips_aligned:
        result='Correct Form'
    else:
        result='Incorrect Form'
    return result

def assess_workout_form_with_model(keypoints):
    prediction = model.predict(preprocessed_keypoints)
    if prediction[0][0] > 0.5:
        return 'Correct Form'
    else:
        return 'Incorrect Form'

class CamerApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        layout = BoxLayout(orientation='vertical')
        self.img = Image()
        layout.add_widget(self.img)
        self.result_label = Label(text='Workout Form: ')
        layout.add_widget(self.result_label)
        btn_exit = Button(text='Exit Camera', size_hint_y=None, height=40)
        btn_exit.bind(on_press=self.stop_camera)
        layout.add_widget(btn_exit)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return layout
    
    def update(self, dt):
        ret, frame=self.capture.read()
        if ret:
            processed_frame=self.process_frame(frame)
            buf1=cv2.flip(processed_frame, 0)
            buf=buf1.tostring()
            image_texture=Texture.create(size=(processed_frame.shape[1], processed_frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img.texture=image_texture
            self.result_label.text=f'Workout Form: {result}'
    
    def process_frame(self, frame):
        keypoints, _ = net(frame)
        if keypoints is not None:
            frame = PoseNet.draw_keypoints(frame, keypoints)
            workout_result = assess_workout_form(keypoints)
            return frame, workout_result
        else:
            return frame, 'Not Detected'
    
    def stop_camera(self, instance):
        self.capture.release()
        App.get_running_app.stop()

def calculate_similarity(keypoints_user, keypoints_referrence):
    distances=np.linalg.norm(keypoints_user-keypoints_referrence, axis=1)
    similarity_score=1/(1+distances.sum())
    return similarity_score

x1_user, y1_user = keypoints['shoulder'][0]
x2_user, y2_user = keypoints['elbow'][0]
x3_user, y3_user = keypoints['wrist'][0]
x4_user, y4_user = keypoints['hip'][0]
x5_user, y5_user = keypoints['knee'][0]
x6_user, y6_user = keypoints['ankle'][0]

shoulder_user = np.array([x1_user, y1_user])
elbow_user = np.array([x2_user, y2_user])
wrist_user = np.array([x3_user, y3_user])
hip_user = np.array([x4_user, y4_user])
knee_user = np.array([x5_user, y5_user])
ankle_user = np.array([x6_user, y6_user])

keypoints_user = np.array([shoulder_user, elbow_user, wrist_user, hip_user, knee_user, ankle_user])

if __name__=='__main__':
    capture_video()
    CamerApp().run()