import sys
import numpy as np
from tensorflow.keras.models import load_model
import data_processing
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal, QThread, QMutex, QMutexLocker
from subprocess import Popen, PIPE
from threading import Thread, Event
from functools import partial

class TrainingThread(QThread):
    update_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.stop_mutex = QMutex()
        self.stop_flag = False
        self.process = None

    def run(self):
        script_path = 'prime.py'
        self.process = Popen(['python', script_path], stdout=PIPE, stderr=PIPE, text=True)
        while not self.should_stop():
            line = self.process.stdout.readline()
            if not line:
                break
            self.update_signal.emit(line.strip())

        self.process.stdout.close()
        self.process.stderr.close()
        self.process = None

    def stop(self):
        with QMutexLocker(self.stop_mutex):
            self.stop_flag = True
        if self.process:
            self.process.terminate()

    def should_stop(self):
        with QMutexLocker(self.stop_mutex):
            return self.stop_flag

class BitcoinPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "Bitcoin Price Prediction"
        self.setGeometry(100, 100, 600, 550)
        self.initUI()

        self.model = load_model('C:/Users/sgrif/Desktop/RaidENT/models/Prime_model.h5')
        self.normalized_historical_data, self.scaler = data_processing.load_historical_data('bitcoin_data.csv')
        self.sequence_length = 30
        self.training_thread = None

    def initUI(self):
        self.setWindowTitle(self.title)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        self.predict_button = QPushButton("Predict Bitcoin Price")
        self.predict_button.clicked.connect(self.start_prediction)
        self.predict_button.setStyleSheet("background-color: #0080FF; color: #FFFFFF;")
        layout.addWidget(self.predict_button)

        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setStyleSheet("background-color: #00FF00; color: #000000;")
        layout.addWidget(self.train_button)

        self.stop_train_button = QPushButton("Stop Training")
        self.stop_train_button.clicked.connect(self.stop_training)
        self.stop_train_button.setStyleSheet("background-color: #FF0000; color: #FFFFFF;")
        layout.addWidget(self.stop_train_button)
        self.stop_train_button.setVisible(False)

        self.status_label = QLabel("Status: Idle")
        self.status_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.status_label.setStyleSheet("color: #000000;")
        layout.addWidget(self.status_label)

        self.live_price_display = QLabel()  # Removed the label text here
        self.live_price_display.setFont(QFont("Arial", 10))
        layout.addWidget(self.live_price_display)

        self.prediction_label = QLabel("Prediction:")
        self.prediction_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(self.prediction_label)

        self.prediction_output = QLabel()
        self.prediction_output.setFont(QFont("Arial", 10))
        layout.addWidget(self.prediction_output)

        self.training_label = QLabel("Training Output:")
        self.training_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(self.training_label)

        self.output_text = QTextEdit()
        self.output_text.setStyleSheet("background-color: #FFFFFF; color: #000000; font-family: 'Arial', sans-serif;")
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        central_widget.setLayout(layout)

    def start_prediction(self):
        real_time_price = data_processing.fetch_real_time_price()
        if real_time_price is None:
            self.prediction_output.setText("Real-time price fetch failed. Exiting.")
            return

        sequences, _ = data_processing.create_sequences_and_targets(self.normalized_historical_data, self.sequence_length)
        input_sequence = sequences[-1:]

        lstm_input_sequence = np.reshape(input_sequence, (input_sequence.shape[0], self.sequence_length, 1))
        conv_input_sequence = np.reshape(input_sequence, (input_sequence.shape[0], self.sequence_length, 1))

        predictions = self.model.predict([lstm_input_sequence, conv_input_sequence])
        denormalized_predictions = data_processing.denormalize_data(predictions, self.scaler)

        predicted_price = int(denormalized_predictions[0][0])
        self.prediction_output.setText(f"Predicted Bitcoin price: {predicted_price}")

    def start_training(self):
        if self.training_thread is None or not self.training_thread.isRunning():
            self.training_thread = TrainingThread()
            self.training_thread.update_signal.connect(self.update_output_text)
            self.training_thread.start()
            self.train_button.setVisible(False)
            self.stop_train_button.setVisible(True)
            self.status_label.setText("Status: Running (Training)")

    def stop_training(self):
        if self.training_thread:
            self.training_thread.stop()
            self.train_button.setVisible(True)
            self.stop_train_button.setVisible(False)
            self.status_label.setText("Status: Idle")

    def update_output_text(self, line):
        self.output_text.append(line.strip())
        self.output_text.moveCursor(QTextCursor.End)  

class LivePriceUpdater(QObject):
    update_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.stop_event = Event()

    def run(self):
        while not self.stop_event.is_set():
            live_price = data_processing.fetch_real_time_price()
            if live_price is not None:
                self.update_signal.emit(f"{live_price}") 
            self.stop_event.wait(60)  # Wait for 60 seconds

    def stop(self):
        self.stop_event.set()

def cleanup(live_price_updater, training_thread):
    live_price_updater.stop()
    if training_thread:
        training_thread.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app_window = BitcoinPredictionApp()

    live_price_updater = LivePriceUpdater()
    live_price_updater.update_signal.connect(app_window.live_price_display.setText)  # Connect to setText
    live_price_thread = Thread(target=live_price_updater.run)
    live_price_thread.start()

    app.aboutToQuit.connect(partial(cleanup, live_price_updater, app_window.training_thread))

    app_window.show()
    sys.exit(app.exec_())











