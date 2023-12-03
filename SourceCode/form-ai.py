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

#VARIABLES
img_dataset_path='C:\Users\shrey\Desktop\Web_projects\miniProject\WorkoutAI\dataset'
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
vid_dataset_path='C:\Users\shrey\Desktop\Web_projects\miniProject\WorkoutAI\dataset'
frames_save_path='C:\Users\shrey\Desktop\Web_projects\miniProject\WorkoutAI\Frames'
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
batch_size = 50
image_height = 128
image_width = 128
net=PoseNet()
result='Correct Form'

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
def capture_video():
    cap=cv2.VideoCapture(0)
    while True:
        ret, frame=cap.read()
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
    # For example, you could check the position of specific keypoints
    # and determine if the workout form is correct or not.
    # Return a string indicating the assessment result.
    # This is a placeholder example, customize it based on your needs.
    if shoulders_aligned and hips_aligned:
        result='Correct Form'
    else:
        result='Incorrect Form'
    return result

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

if __name__=='__main__':
    capture_video()
    CamerApp().run()