import sys
import os
import json
import threading
import queue
import time
from datetime import datetime
import webbrowser
from collections import defaultdict
import csv
from PyQt5.QtWidgets import QFileDialog

import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
from transformers import pipeline
import pygetwindow as gw

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QLabel, QComboBox, QCheckBox,
                             QSpinBox, QMessageBox, QSystemTrayIcon, QMenu, QAction,
                             QSplitter, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, pyqtSlot
from PyQt5.QtGui import QIcon, QColor, QFont

# Configuration
model_path = "C:\\Users\\HpvAn\\Desktop\\vosk-model-small-en-us-0.15\\vosk-model-small-en-us-0.15"  # Updated nested path

DEBUG_MODE = True

class AudioProcessor(QObject):
    update_log = pyqtSignal(str, str)
    action_required = pyqtSignal(str, int)
    participant_updated = pyqtSignal(str, str, str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.audio_queue = queue.Queue()
        self.processing = False
        self.offense_count = defaultdict(int)
        self.participant_actions = {}
        self.keep_running = True
        self.test_mode_active = False

        try:
            self.init_speech_recognition()
            self.init_hate_speech_detection()
            if DEBUG_MODE:
                self.update_log.emit("AudioProcessor initialized successfully", "blue")
        except Exception as e:
            self.update_log.emit(f"Initialization failed: {str(e)}", "red")
            raise

    def init_speech_recognition(self):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found at: {os.path.abspath(model_path)}")

        self.sr_model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.sr_model, 16000)

    def init_hate_speech_detection(self):
        try:
            self.hate_speech_classifier = pipeline(
                "text-classification",
                model="Hate-speech-CNERG/bert-base-uncased-hatexplain",
                device=-1,
                top_k=None
            )
        except Exception as e:
            self.update_log.emit(f"Failed to load hate speech model: {str(e)}", "red")
            raise

    def audio_callback(self, indata, frames, time, status):
        if status:
            self.update_log.emit(f"Audio status: {status}", "orange")
        if self.processing and self.keep_running and not self.test_mode_active:
            self.audio_queue.put(bytes(indata))

    def process_audio(self):
        while self.keep_running:
            try:
                if not self.processing or self.test_mode_active:
                    time.sleep(0.1)
                    continue

                data = self.audio_queue.get(timeout=1)
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '').strip()
                    if text:
                        self.analyze_text(text)
            except queue.Empty:
                continue
            except Exception as e:
                self.update_log.emit(f"Processing error: {str(e)}", "red")

    def analyze_text(self, text):
        self.update_log.emit(f"Recognized: {text}", "black")
        
        try:
            results = self.hate_speech_classifier(text)
            hate_score = next((item['score'] for item in results[0] if item['label'] == 'hate'), 0)
            offensive_score = next((item['score'] for item in results[0] if item['label'] == 'offensive'), 0)

            max_score = max(hate_score, offensive_score)
            if max_score > self.config['sensitivity_threshold']:
                self.handle_offense(text, max_score)
        except Exception as e:
            self.update_log.emit(f"Analysis error: {str(e)}", "red")

    def handle_offense(self, text, score):
        speaker = self.identify_speaker()
        self.offense_count[speaker] += 1
        count = self.offense_count[speaker]

        if score > 0.9 or count > 3:
            severity = 2
            action = "kicked"
        elif score > 0.7 or count > 1:
            severity = 1
            action = "muted"
        else:
            severity = 0
            action = "warned"

        log_msg = f"⚠️ {action.capitalize()} {speaker} (score: {score:.2f}, count: {count})"
        self.update_log.emit(log_msg, "red")
        self.action_required.emit(text, severity)
        
        self.participant_actions[speaker] = {
            'action': action,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'reason': text,
            'score': f"{score:.2f}"
        }
        self.participant_updated.emit(speaker, action, text)

        if severity == 2:
            self.kick_user(speaker)

    def identify_speaker(self):
        speakers = ["Participant 1", "Participant 2", "Participant 3", "Unknown"]
        return speakers[len(self.offense_count) % len(speakers)]

    def kick_user(self, speaker):
        try:
            if not self.test_mode_active:  # Only actually close windows if not in test mode
                closed_windows = []
                for window in gw.getWindowsWithTitle(""):
                    title = window.title.lower()
                    if any(meeting in title for meeting in ['meet', 'zoom', 'teams']):
                        window.close()
                        closed_windows.append(window.title)
                
                if closed_windows:
                    self.update_log.emit(f"Closed windows: {', '.join(closed_windows)}", "red")
                else:
                    self.update_log.emit("No meeting windows found to close", "orange")
            else:
                self.update_log.emit(f"[TEST] Would have kicked {speaker}", "blue")
        except Exception as e:
            self.update_log.emit(f"Error closing windows: {str(e)}", "red")

    def start_processing(self):
        self.processing = True
        self.keep_running = True
        self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.processing_thread.start()

    def stop_processing(self):
        self.processing = False
        self.keep_running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1)

    def set_test_mode(self, active):
        self.test_mode_active = active

class SimulationWorker(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(str, float)

    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.test_phrases = [
            ("That's a stupid idea, you idiot!", 0.95),
            ("People like you shouldn't be allowed here", 0.85),
            ("This is unacceptable behavior", 0.65),
            ("I hate this group so much", 0.92)
        ]

    def run(self):
        try:
            for phrase, score in self.test_phrases:
                self.progress.emit(phrase, score)
                self.processor.handle_offense(phrase, score)
                time.sleep(1)
        except Exception as e:
            print(f"Simulation error: {str(e)}")
        finally:
            self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        try:
            self.config = {
                'sensitivity': 1,
                'sensitivity_threshold': 0.7,
                'default_action': 0,
                'audio_source': 0,
                'offline_mode': True,
                'test_mode': False
            }

            self.init_ui()
            self.init_audio()
            self.init_tray_icon()
            
            if DEBUG_MODE:
                self.update_log("Application initialized successfully", "green")
                
        except Exception as e:
            QMessageBox.critical(None, "Fatal Error", f"Failed to initialize: {str(e)}\n\nPlease check:\n1. Vosk model exists at {model_path}\n2. Audio devices are available")
            sys.exit(1)

    def init_ui(self):
        self.setWindowTitle("Meeting Guardian - Hate Speech Detection")
        self.setGeometry(100, 100, 1000, 700)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout()
        splitter = QSplitter(Qt.Vertical)

        # Control Panel
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Monitoring")
        self.start_btn.clicked.connect(self.start_monitoring)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white;")

        self.stop_btn = QPushButton("Stop Monitoring")
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white;")
        self.stop_btn.setEnabled(False)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.show_settings)
        
        self.test_btn = QPushButton("Test Mode")
        self.test_btn.clicked.connect(self.toggle_test_mode)
        self.test_btn.setStyleSheet("background-color: #ff9800; color: white;")
        
        self.simulate_btn = QPushButton("Simulate Offense")
        self.simulate_btn.clicked.connect(self.simulate_offense)
        self.simulate_btn.setEnabled(False)
        
        self.export_btn = QPushButton("Export Data")
        self.export_btn.clicked.connect(self.export_data)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.settings_btn)
        control_layout.addWidget(self.test_btn)
        control_layout.addWidget(self.simulate_btn)
        control_layout.addWidget(self.export_btn)
        control_panel.setLayout(control_layout)
        
        # Log Display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Courier New", 10))
        
        # Participants Table
        self.participants_table = QTableWidget()
        self.participants_table.setColumnCount(4)
        self.participants_table.setHorizontalHeaderLabels(["Participant", "Action", "Reason", "Score"])
        self.participants_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.participants_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        splitter.addWidget(self.log_display)
        splitter.addWidget(self.participants_table)
        splitter.setSizes([400, 200])
        
        main_layout.addWidget(control_panel)
        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

        self.audio_processor = AudioProcessor(self.config)
        self.audio_processor.update_log.connect(self.update_log)
        self.audio_processor.action_required.connect(self.handle_action)
        self.audio_processor.participant_updated.connect(self.update_participants)

        self.simulation_worker = None

        if not os.path.exists("agreement_accepted"):
            self.show_disclaimer()

    def update_participants(self, name, action, reason):
        row = self.participants_table.rowCount()
        self.participants_table.insertRow(row)
        
        if action == "kicked":
            color = QColor(255, 200, 200)
        elif action == "muted":
            color = QColor(255, 255, 200)
        else:
            color = QColor(200, 255, 200)
            
        for col, value in enumerate([name, action, reason, self.audio_processor.participant_actions[name]['score']]):
            item = QTableWidgetItem(value)
            item.setBackground(color)
            self.participants_table.setItem(row, col, item)

    def toggle_test_mode(self):
        self.config['test_mode'] = not self.config['test_mode']
        self.audio_processor.set_test_mode(self.config['test_mode'])
        self.simulate_btn.setEnabled(self.config['test_mode'])
        
        if self.config['test_mode']:
            self.test_btn.setStyleSheet("background-color: #8bc34a; color: white;")
            self.update_log("Test mode activated", "blue")
        else:
            self.test_btn.setStyleSheet("background-color: #ff9800; color: white;")
            self.update_log("Test mode deactivated", "blue")

    def simulate_offense(self):
        if not self.config['test_mode']:
            self.update_log("Cannot simulate - test mode not active", "orange")
            return
            
        try:
            if self.simulation_worker and self.simulation_worker.isRunning():
                self.update_log("Simulation already running", "orange")
                return

            self.simulation_worker = SimulationWorker(self.audio_processor)
            self.simulation_worker.progress.connect(
                lambda p, s: self.update_log(f"[TEST] Simulating: {p} (score: {s:.2f})", "blue")
            )
            self.simulation_worker.finished.connect(
                lambda: self.update_log("Simulation completed", "green")
            )
            self.simulation_worker.start()
            
        except Exception as e:
            self.update_log(f"Simulation error: {str(e)}", "red")

    def export_data(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "CSV Files (*.csv)", options=options)
            
        if not filename:
            return
            
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Participant Actions Log"])
                writer.writerow(["Name", "Action", "Timestamp", "Reason", "Score"])
                for name, data in self.audio_processor.participant_actions.items():
                    writer.writerow([name, data['action'], data['timestamp'], 
                                   data['reason'], data['score']])
                
                writer.writerow([])
                writer.writerow(["Application Log"])
                writer.writerow(["Timestamp", "Message"])
                log_text = self.log_display.toPlainText()
                for line in log_text.split('\n'):
                    if line.strip():
                        parts = line.split(']')
                        if len(parts) > 1:
                            timestamp = parts[0][1:]
                            message = parts[1].strip()
                            writer.writerow([timestamp, message])
                
            self.update_log(f"Data exported to {filename}", "green")
        except Exception as e:
            self.update_log(f"Export failed: {str(e)}", "red")

    def init_audio(self):
        self.audio_stream = None
        self.sample_rate = 16000
        self.channels = 1
        
        try:
            devices = sd.query_devices()
            if len(devices) == 0:
                raise ValueError("No audio devices found")
            if DEBUG_MODE:
                self.update_log(f"Audio devices available: {len(devices)}", "blue")
        except Exception as e:
            self.update_log(f"Audio device error: {str(e)}", "red")
            QMessageBox.warning(self, "Audio Error", f"Audio initialization failed: {str(e)}")

    def init_tray_icon(self):
        try:
            self.tray_icon = QSystemTrayIcon(self)
            self.tray_icon.setIcon(QIcon("icon.png"))

            tray_menu = QMenu()
            show_action = QAction("Show", self)
            show_action.triggered.connect(self.show)
            tray_menu.addAction(show_action)

            exit_action = QAction("Exit", self)
            exit_action.triggered.connect(self.close)
            tray_menu.addAction(exit_action)

            self.tray_icon.setContextMenu(tray_menu)
            self.tray_icon.show()
        except Exception as e:
            self.update_log(f"Tray icon error: {str(e)}", "orange")

    def show_disclaimer(self):
        disclaimer = """Meeting Guardian Privacy Notice

This application analyzes audio from your meetings in real-time to detect hate speech.

Key points:
- Audio is processed locally and never saved or transmitted
- Only detection results are logged
- You can quit the application at any time

By clicking OK, you agree to this temporary audio processing."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(disclaimer)
        msg.setWindowTitle("Privacy Agreement")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        retval = msg.exec_()
        if retval == QMessageBox.Ok:
            with open("agreement_accepted", "w") as f:
                f.write("1")
        else:
            self.close()

    def show_settings(self):
        self.settings_dialog = SettingsDialog(self.config)
        self.settings_dialog.show()

    def start_monitoring(self):
        try:
            sd.check_input_settings(
                device=sd.default.device[0],
                channels=self.channels,
                dtype='float32'
            )
            
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_processor.audio_callback,
                dtype='float32'
            )
            self.audio_stream.start()

            self.audio_processor.start_processing()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_bar.showMessage("Monitoring active...")
            self.update_log("Monitoring started", "green")

        except sd.PortAudioError as e:
            self.update_log(f"PortAudio Error: {str(e)}", "red")
            QMessageBox.critical(self, "Audio Error", 
                f"Could not access audio device:\n{str(e)}\n\n"
                "Please check your audio settings and try again.")
        except Exception as e:
            self.update_log(f"Monitoring error: {str(e)}", "red")
            QMessageBox.critical(self, "Error", f"Could not start monitoring:\n{str(e)}")

    def stop_monitoring(self):
        try:
            self.audio_processor.stop_processing()
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_bar.showMessage("Monitoring stopped")
            self.update_log("Monitoring stopped", "blue")
        except Exception as e:
            self.update_log(f"Error stopping monitoring: {str(e)}", "red")

    def handle_action(self, text, severity):
        action = self.config['default_action']

        if severity == 0:
            message = f"Warning: Please avoid inappropriate language\n\nOffensive content: {text}"
            self.show_warning(message)
        elif severity == 1:
            if action == 0:
                message = f"Serious Warning: This language violates community guidelines\n\nOffensive content: {text}"
                self.show_warning(message)
            else:
                self.mute_audio()
        else:
            if action < 2:
                self.mute_audio()
            else:
                self.alert_moderator(text)

    def show_warning(self, message):
        self.tray_icon.showMessage(
            "Inappropriate Language Detected",
            message,
            QSystemTrayIcon.Warning,
            5000
        )
        self.update_log("Warning issued to participant", "orange")

    def mute_audio(self):
        self.update_log("System audio muted due to hate speech", "red")
        self.tray_icon.showMessage(
            "Audio Muted",
            "System audio has been muted due to hate speech detection",
            QSystemTrayIcon.Critical,
            5000
        )

    def alert_moderator(self, text):
        self.update_log("Moderator alerted to hate speech", "red")
        self.tray_icon.showMessage(
            "Moderator Alert",
            f"Hate speech detected and moderator notified:\n\n{text}",
            QSystemTrayIcon.Critical,
            5000
        )

    def update_log(self, text, color="black"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        color_code = {
            "black": "#000000",
            "red": "#ff0000",
            "green": "#009900",
            "blue": "#0000ff",
            "orange": "#ff6600"
        }.get(color.lower(), "#000000")

        self.log_display.append(f'<span style="color:{color_code}">[{timestamp}] {text}</span>')
        self.log_display.verticalScrollBar().setValue(self.log_display.verticalScrollBar().maximum())

    def closeEvent(self, event):
        self.stop_monitoring()
        if self.simulation_worker and self.simulation_worker.isRunning():
            self.simulation_worker.quit()
            self.simulation_worker.wait(1000)
        event.accept()

class SettingsDialog(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Settings")
        self.setWindowModality(Qt.ApplicationModal)
        self.setFixedSize(400, 300)

        layout = QVBoxLayout()

        sensitivity_layout = QHBoxLayout()
        sensitivity_layout.addWidget(QLabel("Sensitivity:"))
        self.sensitivity_combo = QComboBox()
        self.sensitivity_combo.addItems(["Low", "Medium", "High"])
        self.sensitivity_combo.setCurrentIndex(self.config['sensitivity'])
        sensitivity_layout.addWidget(self.sensitivity_combo)
        layout.addLayout(sensitivity_layout)

        action_layout = QHBoxLayout()
        action_layout.addWidget(QLabel("Default Action:"))
        self.action_combo = QComboBox()
        self.action_combo.addItems(["Warning", "Mute Audio", "Alert Moderator"])
        self.action_combo.setCurrentIndex(self.config['default_action'])
        action_layout.addWidget(self.action_combo)
        layout.addLayout(action_layout)

        audio_layout = QHBoxLayout()
        audio_layout.addWidget(QLabel("Audio Source:"))
        self.audio_combo = QComboBox()
        self.audio_combo.addItems(["System Default", "Microphone", "Virtual Cable"])
        self.audio_combo.setCurrentIndex(self.config['audio_source'])
        audio_layout.addWidget(self.audio_combo)
        layout.addLayout(audio_layout)

        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)

        self.setLayout(layout)

    def save_settings(self):
        self.config.update({
            'sensitivity': self.sensitivity_combo.currentIndex(),
            'default_action': self.action_combo.currentIndex(),
            'audio_source': self.audio_combo.currentIndex()
        })
        self.close()

def main():
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        app = QApplication(sys.argv)
        app.setStyle('Fusion')

        model_check_path = os.path.join(os.path.dirname(__file__), model_path)
        if not os.path.exists(model_check_path):
            resp = QMessageBox.critical(
                None,
                "Model Required",
                f"Vosk model not found at:\n{os.path.abspath(model_check_path)}\n\n"
                "Please download and extract the model to this location.",
                QMessageBox.Ok
            )
            webbrowser.open("https://alphacephei.com/vosk/models")
            return

        window = MainWindow()
        window.show()
        
        app.main_window = window
        sys.exit(app.exec_())
        
    except Exception as e:
        QMessageBox.critical(None, "Fatal Error", f"Application failed to start:\n{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
