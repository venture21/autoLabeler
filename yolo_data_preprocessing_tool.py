import sys
import os
import cv2
import numpy as np
import copy
import json
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QEvent
from PyQt5.QtGui import *

class BoundingBoxItem:
    def __init__(self, rect, class_id, class_name):
        self.rect = rect
        self.class_id = class_id
        self.class_name = class_name
        
    def copy(self):
        return BoundingBoxItem(self.rect, self.class_id, self.class_name)

# Command classes for undo/redo
class Command:
    def execute(self):
        pass
    
    def undo(self):
        pass

class AddBBoxCommand(Command):
    def __init__(self, canvas, bbox):
        self.canvas = canvas
        self.bbox = bbox
        
    def execute(self):
        self.canvas.bounding_boxes.append(self.bbox)
        
    def undo(self):
        if self.bbox in self.canvas.bounding_boxes:
            self.canvas.bounding_boxes.remove(self.bbox)

class DeleteBBoxCommand(Command):
    def __init__(self, canvas, bbox, index):
        self.canvas = canvas
        self.bbox = bbox
        self.index = index
        
    def execute(self):
        if self.bbox in self.canvas.bounding_boxes:
            self.canvas.bounding_boxes.remove(self.bbox)
            
    def undo(self):
        self.canvas.bounding_boxes.insert(self.index, self.bbox)

class MoveBBoxCommand(Command):
    def __init__(self, canvas, bbox, old_rect, new_rect):
        self.canvas = canvas
        self.bbox = bbox
        self.old_rect = old_rect
        self.new_rect = new_rect
        
    def execute(self):
        self.bbox.rect = self.new_rect
        
    def undo(self):
        self.bbox.rect = self.old_rect

class ResizeBBoxCommand(Command):
    def __init__(self, canvas, bbox, old_rect, new_rect):
        self.canvas = canvas
        self.bbox = bbox
        self.old_rect = old_rect
        self.new_rect = new_rect
        
    def execute(self):
        self.bbox.rect = self.new_rect
        
    def undo(self):
        self.bbox.rect = self.old_rect

class ImageCanvas(QLabel):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("border: 1px solid black;")
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)  # Set minimum size
        
        self.image = None
        self.original_image = None
        self.display_image = None
        self.bounding_boxes = []
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.selected_class_id = 0
        self.scale_factor = 1.0
        self._updating = False  # Flag to prevent recursive updates
        
        # For bbox dragging
        self.dragging_bbox = None
        self.drag_start_pos = None
        self.drag_original_rect = None
        
        # For bbox resizing
        self.resizing_bbox = None
        self.resize_start_pos = None
        self.resize_original_rect = None
        self.resize_handle = None  # 'left', 'right', 'top', 'bottom', 'tl', 'tr', 'bl', 'br'
        
        # Undo/Redo stacks
        self.undo_stack = []
        self.redo_stack = []
        
        # Define colors for different classes (BGR format for OpenCV)
        self.class_colors = [
            (0, 255, 0),      # Green
            (255, 0, 0),      # Blue
            (0, 0, 255),      # Red
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
            (128, 255, 0),    # Light Green
            (255, 128, 0),    # Light Blue
            (128, 0, 255),    # Purple
            (255, 128, 128),  # Light Red
        ]
        
    def set_image(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            return False
        self.original_image = self.image.copy()
        self.display_image = self.original_image.copy()
        self.update_display()
        return True
        
    def update_display(self):
        if self.original_image is None or self._updating:
            return
            
        self._updating = True
        
        # Always work from original image
        temp_image = self.original_image.copy()
        
        # Draw bounding boxes
        for bbox in self.bounding_boxes:
            # Get color for this class
            color = self.class_colors[bbox.class_id % len(self.class_colors)]
            
            cv2.rectangle(temp_image, 
                         (int(bbox.rect[0]), int(bbox.rect[1])), 
                         (int(bbox.rect[2]), int(bbox.rect[3])), 
                         color, 2)
            cv2.putText(temp_image, bbox.class_name, 
                       (int(bbox.rect[0]), int(bbox.rect[1]-5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw current drawing box
        if self.drawing and self.start_point and self.end_point:
            # Get color for currently selected class
            current_color = self.class_colors[self.selected_class_id % len(self.class_colors)]
            cv2.rectangle(temp_image, self.start_point, self.end_point, current_color, 2)
            
        # Convert to QPixmap and display
        height, width, channel = temp_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(temp_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Scale to fit widget while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        widget_size = self.size()
        
        # Calculate scale factor to fit image within widget
        if widget_size.width() > 0 and widget_size.height() > 0:
            # Get available space (with some margin)
            available_width = widget_size.width() - 20
            available_height = widget_size.height() - 20
            
            scale_w = available_width / width
            scale_h = available_height / height
            self.scale_factor = min(scale_w, scale_h)
            
            # Limit scale factor
            if self.scale_factor > 1.0:
                self.scale_factor = 1.0  # Don't scale up beyond original size
        else:
            self.scale_factor = 1.0
        
        # Scale pixmap
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        scaled_pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.setPixmap(scaled_pixmap)
        self._updating = False
        
    def mousePressEvent(self, event):
        if self.original_image is None:
            return
            
        # Convert widget coordinates to image coordinates
        pos = self.get_image_coordinates(event.pos())
        if pos is None:
            return
            
        # Check modifier keys
        modifiers = QApplication.keyboardModifiers()
        
        if event.button() == Qt.LeftButton:
            if modifiers == Qt.ControlModifier:
                # Ctrl + Click: Move bounding box
                for bbox in self.bounding_boxes:
                    if (bbox.rect[0] <= pos[0] <= bbox.rect[2] and 
                        bbox.rect[1] <= pos[1] <= bbox.rect[3]):
                        self.dragging_bbox = bbox
                        self.drag_start_pos = pos
                        self.drag_original_rect = copy.deepcopy(bbox.rect)
                        break
            elif modifiers == Qt.AltModifier:
                # Alt + Click: Resize bounding box
                closest_bbox, handle = self.find_closest_bbox_for_resize(pos)
                if closest_bbox and handle:
                    self.resizing_bbox = closest_bbox
                    self.resize_start_pos = pos
                    self.resize_original_rect = copy.deepcopy(closest_bbox.rect)
                    self.resize_handle = handle
            else:
                # Normal drawing mode
                self.drawing = True
                self.start_point = pos
                self.end_point = pos
        elif event.button() == Qt.RightButton:
            # Check if click is inside any bounding box
            for i, bbox in enumerate(self.bounding_boxes):
                if (bbox.rect[0] <= pos[0] <= bbox.rect[2] and 
                    bbox.rect[1] <= pos[1] <= bbox.rect[3]):
                    reply = QMessageBox.question(self, '바운딩 박스 삭제', 
                                               '바운딩 박스를 삭제할까요?',
                                               QMessageBox.Yes | QMessageBox.No)
                    if reply == QMessageBox.Yes:
                        cmd = DeleteBBoxCommand(self, bbox, i)
                        cmd.execute()
                        self.execute_command(cmd)
                        self.update_display()
                    break
                    
    def mouseMoveEvent(self, event):
        if self.original_image is None:
            return
            
        pos = self.get_image_coordinates(event.pos())
        if pos is None:
            return
            
        if self.drawing:
            self.end_point = pos
            self.update_display()
        elif self.dragging_bbox and self.drag_start_pos:
            # Calculate movement
            dx = pos[0] - self.drag_start_pos[0]
            dy = pos[1] - self.drag_start_pos[1]
            
            # Update bbox position
            new_rect = (
                self.drag_original_rect[0] + dx,
                self.drag_original_rect[1] + dy,
                self.drag_original_rect[2] + dx,
                self.drag_original_rect[3] + dy
            )
            
            # Check bounds
            h, w = self.original_image.shape[:2]
            if (0 <= new_rect[0] and new_rect[2] <= w and 
                0 <= new_rect[1] and new_rect[3] <= h):
                self.dragging_bbox.rect = new_rect
                self.update_display()
        elif self.resizing_bbox and self.resize_start_pos and self.resize_handle:
            # Calculate resize
            dx = pos[0] - self.resize_start_pos[0]
            dy = pos[1] - self.resize_start_pos[1]
            
            x1, y1, x2, y2 = self.resize_original_rect
            
            # Apply resize based on handle
            if self.resize_handle == 'left':
                new_rect = (x1 + dx, y1, x2, y2)
            elif self.resize_handle == 'right':
                new_rect = (x1, y1, x2 + dx, y2)
            elif self.resize_handle == 'top':
                new_rect = (x1, y1 + dy, x2, y2)
            elif self.resize_handle == 'bottom':
                new_rect = (x1, y1, x2, y2 + dy)
            elif self.resize_handle == 'tl':  # top-left
                new_rect = (x1 + dx, y1 + dy, x2, y2)
            elif self.resize_handle == 'tr':  # top-right
                new_rect = (x1, y1 + dy, x2 + dx, y2)
            elif self.resize_handle == 'bl':  # bottom-left
                new_rect = (x1 + dx, y1, x2, y2 + dy)
            elif self.resize_handle == 'br':  # bottom-right
                new_rect = (x1, y1, x2 + dx, y2 + dy)
            else:
                return
                
            # Ensure valid rectangle (left < right, top < bottom)
            left = min(new_rect[0], new_rect[2])
            top = min(new_rect[1], new_rect[3])
            right = max(new_rect[0], new_rect[2])
            bottom = max(new_rect[1], new_rect[3])
            
            # Check minimum size and bounds
            h, w = self.original_image.shape[:2]
            if (right - left >= 5 and bottom - top >= 5 and 
                0 <= left and right <= w and 0 <= top and bottom <= h):
                self.resizing_bbox.rect = (left, top, right, bottom)
                self.update_display()
        else:
            # Update cursor based on position
            self.update_cursor(pos)
                
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.drawing:
                self.drawing = False
                if self.start_point and self.end_point:
                    # Normalize coordinates
                    x1, y1 = self.start_point
                    x2, y2 = self.end_point
                    rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                    
                    # Check minimum size (5 pixels)
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    
                    if width >= 5 and height >= 5:
                        # Get selected class from parent
                        parent = self.parent()
                        while parent and not hasattr(parent, 'get_selected_class'):
                            parent = parent.parent()
                            
                        if parent:
                            class_id, class_name = parent.get_selected_class()
                            self.selected_class_id = class_id  # Update selected class id
                            bbox = BoundingBoxItem(rect, class_id, class_name)
                            cmd = AddBBoxCommand(self, bbox)
                            cmd.execute()
                            self.execute_command(cmd)
                        
                self.update_display()
                
            elif self.dragging_bbox and self.drag_original_rect:
                # Finish dragging
                if self.dragging_bbox.rect != self.drag_original_rect:
                    cmd = MoveBBoxCommand(self, self.dragging_bbox, 
                                        self.drag_original_rect, 
                                        copy.deepcopy(self.dragging_bbox.rect))
                    self.execute_command(cmd)
                    
                self.dragging_bbox = None
                self.drag_start_pos = None
                self.drag_original_rect = None
                self.update_display()
                
            elif self.resizing_bbox and self.resize_original_rect:
                # Finish resizing
                if self.resizing_bbox.rect != self.resize_original_rect:
                    cmd = ResizeBBoxCommand(self, self.resizing_bbox,
                                          self.resize_original_rect,
                                          copy.deepcopy(self.resizing_bbox.rect))
                    self.execute_command(cmd)
                    
                self.resizing_bbox = None
                self.resize_start_pos = None
                self.resize_original_rect = None
                self.resize_handle = None
                self.update_display()
            
    def get_image_coordinates(self, widget_pos):
        if self.original_image is None:
            return None
            
        # Get pixmap
        pixmap = self.pixmap()
        if not pixmap:
            return None
            
        # Calculate actual displayed image size
        height, width = self.original_image.shape[:2]
        display_width = int(width * self.scale_factor)
        display_height = int(height * self.scale_factor)
        
        # Calculate offset (centered image)
        widget_rect = self.rect()
        x_offset = (widget_rect.width() - display_width) // 2
        y_offset = (widget_rect.height() - display_height) // 2
        
        # Convert widget coordinates to image coordinates
        x = (widget_pos.x() - x_offset) / self.scale_factor
        y = (widget_pos.y() - y_offset) / self.scale_factor
        
        # Check bounds
        if 0 <= x < width and 0 <= y < height:
            return (int(x), int(y))
        return None
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_image is not None:
            self.update_display()
            
    def leaveEvent(self, event):
        """Reset cursor when mouse leaves the widget"""
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)
            
    def execute_command(self, command):
        self.undo_stack.append(command)
        self.redo_stack.clear()  # Clear redo stack when new action is performed
        
        # Mark annotations as changed and auto-save
        parent = self.parent()
        while parent and not hasattr(parent, 'mark_annotations_changed'):
            parent = parent.parent()
        if parent:
            parent.mark_annotations_changed()
        
    def undo(self):
        if self.undo_stack:
            command = self.undo_stack.pop()
            command.undo()
            self.redo_stack.append(command)
            self.update_display()
            
            # Mark annotations as changed and auto-save
            parent = self.parent()
            while parent and not hasattr(parent, 'mark_annotations_changed'):
                parent = parent.parent()
            if parent:
                parent.mark_annotations_changed()
            
    def redo(self):
        if self.redo_stack:
            command = self.redo_stack.pop()
            command.execute()
            self.undo_stack.append(command)
            self.update_display()
            
            # Mark annotations as changed and auto-save
            parent = self.parent()
            while parent and not hasattr(parent, 'mark_annotations_changed'):
                parent = parent.parent()
            if parent:
                parent.mark_annotations_changed()
            
    def clear_history(self):
        self.undo_stack.clear()
        self.redo_stack.clear()
        
    def get_resize_handle(self, pos, bbox):
        """Get resize handle for the given position and bounding box"""
        handle_size = 8  # Size of resize handle area
        x1, y1, x2, y2 = bbox.rect
        px, py = pos
        
        # Check corners first
        if abs(px - x1) <= handle_size and abs(py - y1) <= handle_size:
            return 'tl'  # top-left
        elif abs(px - x2) <= handle_size and abs(py - y1) <= handle_size:
            return 'tr'  # top-right
        elif abs(px - x1) <= handle_size and abs(py - y2) <= handle_size:
            return 'bl'  # bottom-left
        elif abs(px - x2) <= handle_size and abs(py - y2) <= handle_size:
            return 'br'  # bottom-right
        
        # Check edges
        elif abs(px - x1) <= handle_size and y1 <= py <= y2:
            return 'left'
        elif abs(px - x2) <= handle_size and y1 <= py <= y2:
            return 'right'
        elif abs(py - y1) <= handle_size and x1 <= px <= x2:
            return 'top'
        elif abs(py - y2) <= handle_size and x1 <= px <= x2:
            return 'bottom'
            
        return None
        
    def find_closest_bbox_for_resize(self, pos):
        """Find the closest bounding box for resizing"""
        closest_bbox = None
        closest_handle = None
        min_distance = float('inf')
        
        for bbox in self.bounding_boxes:
            handle = self.get_resize_handle(pos, bbox)
            if handle:
                # Calculate distance to bbox center for tie-breaking
                x1, y1, x2, y2 = bbox.rect
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                distance = ((pos[0] - center_x) ** 2 + (pos[1] - center_y) ** 2) ** 0.5
                
                if distance < min_distance:
                    min_distance = distance
                    closest_bbox = bbox
                    closest_handle = handle
                    
        return closest_bbox, closest_handle
        
    def update_cursor(self, pos):
        """Update cursor based on mouse position"""
        modifiers = QApplication.keyboardModifiers()
        
        if modifiers == Qt.AltModifier:
            # Check if mouse is near any bbox edge for resizing
            closest_bbox, handle = self.find_closest_bbox_for_resize(pos)
            if handle:
                # Set appropriate resize cursor
                if handle in ['left', 'right']:
                    self.setCursor(Qt.SizeHorCursor)
                elif handle in ['top', 'bottom']:
                    self.setCursor(Qt.SizeVerCursor)
                elif handle in ['tl', 'br']:
                    self.setCursor(Qt.SizeFDiagCursor)
                elif handle in ['tr', 'bl']:
                    self.setCursor(Qt.SizeBDiagCursor)
                return
        elif modifiers == Qt.ControlModifier:
            # Check if mouse is inside any bbox for moving
            for bbox in self.bounding_boxes:
                if (bbox.rect[0] <= pos[0] <= bbox.rect[2] and 
                    bbox.rect[1] <= pos[1] <= bbox.rect[3]):
                    self.setCursor(Qt.SizeAllCursor)
                    return
                    
        # Default cursor
        self.setCursor(Qt.CrossCursor)

class DataAugmentationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("데이터 증식 옵션")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Augmentation options
        self.flip_horizontal = QCheckBox("이미지 좌우 대칭")
        layout.addWidget(self.flip_horizontal)
        
        self.flip_vertical = QCheckBox("이미지 수직 대칭")
        layout.addWidget(self.flip_vertical)
        
        # Rotation
        rotation_layout = QHBoxLayout()
        self.rotate_check = QCheckBox("이미지 회전")
        rotation_layout.addWidget(self.rotate_check)
        self.rotate_angle = QSpinBox()
        self.rotate_angle.setRange(-180, 180)
        self.rotate_angle.setValue(0)
        self.rotate_angle.setSuffix("°")
        rotation_layout.addWidget(self.rotate_angle)
        rotation_layout.addStretch()
        layout.addLayout(rotation_layout)
        
        # Scale
        scale_layout = QHBoxLayout()
        self.scale_check = QCheckBox("이미지 스케일 변경")
        scale_layout.addWidget(self.scale_check)
        self.scale_factor = QDoubleSpinBox()
        self.scale_factor.setRange(0.5, 2.0)
        self.scale_factor.setSingleStep(0.1)
        self.scale_factor.setValue(1.0)
        scale_layout.addWidget(self.scale_factor)
        scale_layout.addStretch()
        layout.addLayout(scale_layout)
        
        # Apply button
        self.apply_button = QPushButton("증식 적용")
        self.apply_button.setStyleSheet("padding: 10px; font-size: 14px;")
        layout.addWidget(self.apply_button)
        
        layout.addStretch()
        self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_index = -1
        self.image_files = []
        self.current_directory = ""
        self.label_directory = ""  # Separate directory for labels
        self.classes = []
        self.selected_class_index = 0
        self.config_file = "config.txt"
        self.loading_initial_image = True  # Flag to prevent auto-save during startup
        self.annotations_changed = False  # Flag to track if annotations have changed
        
        self.load_classes()
        self.load_config()  # Load saved config
        self.init_ui()
        
        # Load images from saved directory if available
        if self.current_directory and os.path.exists(self.current_directory):
            self.load_image_files()
            
        # Update UI labels
        self.update_directory_labels()
        
        # Allow auto-save after initial loading is complete
        self.loading_initial_image = False
        
    def mark_annotations_changed(self):
        """Mark that annotations have been changed and save them"""
        self.annotations_changed = True
        if not self.loading_initial_image:
            self.save_annotations()
        
    def load_classes(self):
        try:
            with open('classes.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.classes.append(line)
        except:
            self.classes = ["car", "bus", "truck"]
            
    def load_config(self):
        """Load saved configuration from config.txt"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.current_directory = config.get('image_dir', '')
                    self.label_directory = config.get('label_dir', '')
        except Exception as e:
            print(f"Failed to load config: {e}")
            
    def save_config(self):
        """Save current configuration to config.txt"""
        try:
            config = {
                'image_dir': self.current_directory,
                'label_dir': self.label_directory
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Failed to save config: {e}")
            
    def init_ui(self):
        self.setWindowTitle("YOLO v8 데이터 전처리 및 증식 도구")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left menu bar
        left_menu = QVBoxLayout()
        
        self.open_dir_btn = QPushButton("Image Dir")
        self.open_dir_btn.clicked.connect(self.open_directory)
        left_menu.addWidget(self.open_dir_btn)
        
        # Show current image directory
        self.image_dir_label = QLabel("No image dir")
        self.image_dir_label.setStyleSheet("font-size: 16px; color: gray; margin: 2px;")
        self.image_dir_label.setWordWrap(True)
        left_menu.addWidget(self.image_dir_label)
        
        self.label_dir_btn = QPushButton("Label Dir")
        self.label_dir_btn.clicked.connect(self.open_label_directory)
        left_menu.addWidget(self.label_dir_btn)
        
        # Show current label directory
        self.label_dir_label = QLabel("No label dir")
        self.label_dir_label.setStyleSheet("font-size: 16px; color: gray; margin: 2px;")
        self.label_dir_label.setWordWrap(True)
        left_menu.addWidget(self.label_dir_label)
        
        self.prev_img_btn = QPushButton("Prev Image")
        self.prev_img_btn.clicked.connect(self.prev_image)
        left_menu.addWidget(self.prev_img_btn)
        
        self.next_img_btn = QPushButton("Next Image")
        self.next_img_btn.clicked.connect(self.next_image)
        left_menu.addWidget(self.next_img_btn)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_annotations)
        left_menu.addWidget(self.save_btn)
        
        self.data_aug_btn = QPushButton("Data Aug")
        self.data_aug_btn.setCheckable(True)
        self.data_aug_btn.clicked.connect(self.toggle_data_augmentation)
        left_menu.addWidget(self.data_aug_btn)
        
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo_action)
        left_menu.addWidget(self.undo_btn)
        
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.clicked.connect(self.redo_action)
        left_menu.addWidget(self.redo_btn)
        
        left_menu.addStretch()
        main_layout.addLayout(left_menu)
        
        # Create horizontal splitter for center and right panels
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Center area - stacked widget for image canvas and augmentation options
        self.stacked_widget = QStackedWidget()
        
        # Image canvas
        self.image_canvas = ImageCanvas()
        self.stacked_widget.addWidget(self.image_canvas)
        
        # Data augmentation widget
        self.aug_widget = DataAugmentationWidget()
        self.aug_widget.apply_button.clicked.connect(self.apply_augmentation)
        self.stacked_widget.addWidget(self.aug_widget)
        
        # Right panel widget
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        # Class selection
        class_group = QGroupBox("Box Labels")
        class_layout = QVBoxLayout()
        
        self.class_radio_buttons = []
        for i, class_name in enumerate(self.classes):
            radio = QRadioButton(class_name)
            if i == 0:
                radio.setChecked(True)
            radio.toggled.connect(lambda checked, idx=i: self.select_class(idx) if checked else None)
            self.class_radio_buttons.append(radio)
            class_layout.addWidget(radio)
            
        class_group.setLayout(class_layout)
        right_layout.addWidget(class_group)
        
        # File list
        file_group = QGroupBox("File List")
        file_layout = QVBoxLayout()
        
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.file_selected)
        self.file_list.installEventFilter(self)  # Install event filter for keyboard events
        self.file_list.setMinimumWidth(200)  # Set minimum width
        file_layout.addWidget(self.file_list)
        
        file_group.setLayout(file_layout)
        right_layout.addWidget(file_group)
        
        # Add widgets to splitter
        self.main_splitter.addWidget(self.stacked_widget)
        self.main_splitter.addWidget(right_widget)
        
        # Set initial splitter sizes (70% for image, 30% for right panel)
        self.main_splitter.setStretchFactor(0, 7)
        self.main_splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(self.main_splitter)
        
    def open_directory(self):
        # Start from previously saved directory if available
        start_dir = self.current_directory if self.current_directory else ""
        directory = QFileDialog.getExistingDirectory(self, "Select Image Directory", start_dir)
        if directory:
            # Save current annotations if they were changed
            if self.annotations_changed and not self.loading_initial_image:
                self.save_annotations()
                
            self.loading_initial_image = True  # Prevent auto-save when loading new directory
            self.current_directory = directory
            # If label directory is not set, use image directory as default
            if not self.label_directory:
                self.label_directory = directory
            self.save_config()  # Save config when directory is selected
            self.update_directory_labels()  # Update UI labels
            self.load_image_files()
            self.loading_initial_image = False  # Re-enable auto-save
            self.annotations_changed = False  # Reset change flag
            
    def open_label_directory(self):
        # Start from previously saved directory if available
        start_dir = self.label_directory if self.label_directory else self.current_directory
        directory = QFileDialog.getExistingDirectory(self, "Select Label Directory", start_dir)
        if directory:
            # Save current annotations if they were changed
            if self.annotations_changed and not self.loading_initial_image:
                self.save_annotations()
                
            self.label_directory = directory
            self.save_config()  # Save config when directory is selected
            self.update_directory_labels()  # Update UI labels
            
            # Reload current image annotations from new label directory
            if self.current_image_index >= 0:
                self.load_annotations()
            self.annotations_changed = False  # Reset change flag
            
    def update_directory_labels(self):
        """Update the directory labels in the UI"""
        if hasattr(self, 'image_dir_label'):
            if self.current_directory:
                # Show only the last part of the path to save space
                dir_name = os.path.basename(self.current_directory) or self.current_directory
                self.image_dir_label.setText(dir_name)
                self.image_dir_label.setToolTip(self.current_directory)  # Full path in tooltip
            else:
                self.image_dir_label.setText("No image dir")
                self.image_dir_label.setToolTip("")
                
        if hasattr(self, 'label_dir_label'):
            if self.label_directory:
                # Show only the last part of the path to save space
                dir_name = os.path.basename(self.label_directory) or self.label_directory
                self.label_dir_label.setText(dir_name)
                self.label_dir_label.setToolTip(self.label_directory)  # Full path in tooltip
            else:
                self.label_dir_label.setText("No label dir")
                self.label_dir_label.setToolTip("")
            
    def load_image_files(self):
        self.image_files = []
        self.file_list.clear()
        
        if not self.current_directory:
            return
            
        # Load image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        for file in os.listdir(self.current_directory):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                self.image_files.append(file)
                self.file_list.addItem(file)
                
        if self.image_files:
            self.current_image_index = 0
            self.load_current_image()
            
    def load_current_image(self):
        if 0 <= self.current_image_index < len(self.image_files):
            image_file = self.image_files[self.current_image_index]
            image_path = os.path.join(self.current_directory, image_file)
            
            if self.image_canvas.set_image(image_path):
                # Temporarily disconnect signal to prevent auto-save during programmatic selection
                self.file_list.itemClicked.disconnect()
                self.file_list.setCurrentRow(self.current_image_index)
                self.file_list.itemClicked.connect(self.file_selected)
                
                self.image_canvas.clear_history()  # Clear undo/redo history when loading new image
                self.load_annotations()
                self.annotations_changed = False  # Reset change flag for new image
                
    def load_annotations(self):
        if not self.current_directory or self.current_image_index < 0:
            return
            
        self.image_canvas.bounding_boxes = []
        
        # Load YOLO format annotations from label directory
        image_file = self.image_files[self.current_image_index]
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(self.label_directory, label_file)
        
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    content = f.read()
                    
                    if not content.strip():
                        self.image_canvas.update_display()
                        return
                        
                    h, w = self.image_canvas.original_image.shape[:2]
                    
                    for line in content.strip().split('\n'):
                        parts = line.strip().split()
                        
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:5])
                            
                            # Convert from YOLO format to pixel coordinates
                            x1 = int((cx - bw/2) * w)
                            y1 = int((cy - bh/2) * h)
                            x2 = int((cx + bw/2) * w)
                            y2 = int((cy + bh/2) * h)
                            
                            class_name = self.classes[class_id] if class_id < len(self.classes) else "unknown"
                            bbox = BoundingBoxItem((x1, y1, x2, y2), class_id, class_name)
                            self.image_canvas.bounding_boxes.append(bbox)
                            
            except Exception as e:
                print(f"Error reading annotation file: {e}")
                        
        self.image_canvas.update_display()
        
    def save_annotations(self):
        if not self.current_directory or self.current_image_index < 0:
            return
            
        # Create label directory if it doesn't exist
        if self.label_directory and not os.path.exists(self.label_directory):
            os.makedirs(self.label_directory, exist_ok=True)
            
        image_file = self.image_files[self.current_image_index]
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(self.label_directory, label_file)
        
        h, w = self.image_canvas.original_image.shape[:2]
        
        with open(label_path, 'w') as f:
            for bbox in self.image_canvas.bounding_boxes:
                # Check minimum size before saving
                width = bbox.rect[2] - bbox.rect[0]
                height = bbox.rect[3] - bbox.rect[1]
                
                if width >= 5 and height >= 5:
                    # Convert to YOLO format
                    cx = (bbox.rect[0] + bbox.rect[2]) / 2 / w
                    cy = (bbox.rect[1] + bbox.rect[3]) / 2 / h
                    bw = (bbox.rect[2] - bbox.rect[0]) / w
                    bh = (bbox.rect[3] - bbox.rect[1]) / h
                    
                    f.write(f"{bbox.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                
    def next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            # Only save if annotations were actually changed
            if self.annotations_changed:
                self.save_annotations()
            self.current_image_index += 1
            self.load_current_image()
            
    def prev_image(self):
        if self.image_files and self.current_image_index > 0:
            # Only save if annotations were actually changed
            if self.annotations_changed:
                self.save_annotations()
            self.current_image_index -= 1
            self.load_current_image()
            
    def file_selected(self, item):
        index = self.file_list.row(item)
        if index != self.current_image_index:
            # Only save if annotations were actually changed
            if self.annotations_changed and not self.loading_initial_image:
                self.save_annotations()
            self.current_image_index = index
            self.load_current_image()
            
    def select_class(self, index):
        self.selected_class_index = index
        
    def get_selected_class(self):
        return (self.selected_class_index, self.classes[self.selected_class_index])
        
    def toggle_data_augmentation(self, checked):
        if checked:
            self.stacked_widget.setCurrentIndex(1)
            self.data_aug_btn.setText("Show Image")
        else:
            self.stacked_widget.setCurrentIndex(0)
            self.data_aug_btn.setText("Data Aug")
            
    def apply_augmentation(self):
        if not self.current_directory or not self.image_files:
            QMessageBox.warning(self, "경고", "먼저 디렉토리를 선택하세요.")
            return
            
        # Create augmented directory structure
        aug_dir = os.path.join(self.current_directory, "augmented")
        aug_images_dir = os.path.join(aug_dir, "images")
        aug_labels_dir = os.path.join(aug_dir, "labels")
        
        os.makedirs(aug_images_dir, exist_ok=True)
        os.makedirs(aug_labels_dir, exist_ok=True)
        
        print(f"DEBUG: Aug directories created:")
        print(f"  Images: {aug_images_dir}")
        print(f"  Labels: {aug_labels_dir}")
        
        # Process each image
        progress = QProgressDialog("증식 처리 중...", "취소", 0, len(self.image_files), self)
        progress.setWindowModality(Qt.WindowModal)
        
        for i, image_file in enumerate(self.image_files):
            if progress.wasCanceled():
                break
                
            progress.setValue(i)
            print(f"DEBUG: Processing {image_file}")
            
            # Load image and annotations
            image_path = os.path.join(self.current_directory, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"DEBUG: Failed to load image: {image_path}")
                continue
                
            base_name = os.path.splitext(image_file)[0]
            
            # Load annotations from label directory (not current directory)
            label_file = base_name + '.txt'
            label_path = os.path.join(self.label_directory, label_file)
            annotations = []
            
            print(f"DEBUG: Looking for label file: {label_path}")
            print(f"DEBUG: Label file exists: {os.path.exists(label_path)}")
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    annotations = f.readlines()
                    print(f"DEBUG: Loaded {len(annotations)} annotations: {annotations}")
            else:
                print(f"DEBUG: No label file found, using empty annotations")
                    
            # Apply augmentations
            if self.aug_widget.flip_horizontal.isChecked():
                print("DEBUG: Applying horizontal flip")
                aug_image = cv2.flip(image, 1)
                aug_annotations = self.flip_annotations_horizontal(annotations)
                print(f"DEBUG: Horizontal flip annotations: {aug_annotations}")
                
                # Save to separate directories
                cv2.imwrite(os.path.join(aug_images_dir, f"{base_name}_flip_h.jpg"), aug_image)
                with open(os.path.join(aug_labels_dir, f"{base_name}_flip_h.txt"), 'w') as f:
                    f.writelines(aug_annotations)
                    
            if self.aug_widget.flip_vertical.isChecked():
                print("DEBUG: Applying vertical flip")
                aug_image = cv2.flip(image, 0)
                aug_annotations = self.flip_annotations_vertical(annotations)
                print(f"DEBUG: Vertical flip annotations: {aug_annotations}")
                
                # Save to separate directories
                cv2.imwrite(os.path.join(aug_images_dir, f"{base_name}_flip_v.jpg"), aug_image)
                with open(os.path.join(aug_labels_dir, f"{base_name}_flip_v.txt"), 'w') as f:
                    f.writelines(aug_annotations)
                    
            if self.aug_widget.rotate_check.isChecked():
                angle = self.aug_widget.rotate_angle.value()
                if angle != 0:
                    print(f"DEBUG: Applying rotation {angle} degrees")
                    aug_image, aug_annotations = self.rotate_image_and_annotations(image, annotations, angle)
                    print(f"DEBUG: Rotation annotations: {aug_annotations}")
                    
                    # Save to separate directories
                    cv2.imwrite(os.path.join(aug_images_dir, f"{base_name}_rot_{angle}.jpg"), aug_image)
                    with open(os.path.join(aug_labels_dir, f"{base_name}_rot_{angle}.txt"), 'w') as f:
                        f.writelines(aug_annotations)
                        
            if self.aug_widget.scale_check.isChecked():
                scale = self.aug_widget.scale_factor.value()
                if scale != 1.0:
                    print(f"DEBUG: Applying scale {scale}")
                    aug_image = self.scale_image(image, scale)
                    print(f"DEBUG: Scale annotations (unchanged): {annotations}")
                    
                    # Save to separate directories
                    cv2.imwrite(os.path.join(aug_images_dir, f"{base_name}_scale_{scale}.jpg"), aug_image)
                    with open(os.path.join(aug_labels_dir, f"{base_name}_scale_{scale}.txt"), 'w') as f:
                        f.writelines(annotations)  # Annotations remain same for scaling
                        
        progress.setValue(len(self.image_files))
        QMessageBox.information(self, "완료", 
                              f"증식이 완료되었습니다.\n\n"
                              f"저장 위치:\n"
                              f"• 이미지: {aug_images_dir}\n"
                              f"• 레이블: {aug_labels_dir}")
        
    def flip_annotations_horizontal(self, annotations):
        flipped = []
        for ann in annotations:
            parts = ann.strip().split()
            if len(parts) >= 5:
                class_id = parts[0]
                cx = 1.0 - float(parts[1])
                cy, w, h = parts[2:5]
                flipped.append(f"{class_id} {cx} {cy} {w} {h}\n")
        return flipped
        
    def flip_annotations_vertical(self, annotations):
        flipped = []
        for ann in annotations:
            parts = ann.strip().split()
            if len(parts) >= 5:
                class_id = parts[0]
                cx = parts[1]
                cy = 1.0 - float(parts[2])
                w, h = parts[3:5]
                flipped.append(f"{class_id} {cx} {cy} {w} {h}\n")
        return flipped
        
    def rotate_image_and_annotations(self, image, annotations, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Rotate image
        rotated = cv2.warpAffine(image, M, (new_w, new_h))
        
        # Rotate annotations
        rotated_annotations = []
        for ann in annotations:
            parts = ann.strip().split()
            if len(parts) >= 5:
                class_id = parts[0]
                cx, cy, bw, bh = map(float, parts[1:5])
                
                # Convert to pixel coordinates
                px = cx * w
                py = cy * h
                
                # Apply rotation
                new_px = M[0, 0] * px + M[0, 1] * py + M[0, 2]
                new_py = M[1, 0] * px + M[1, 1] * py + M[1, 2]
                
                # Convert back to normalized coordinates
                new_cx = new_px / new_w
                new_cy = new_py / new_h
                
                # Keep within bounds
                new_cx = max(0, min(1, new_cx))
                new_cy = max(0, min(1, new_cy))
                
                rotated_annotations.append(f"{class_id} {new_cx:.6f} {new_cy:.6f} {bw:.6f} {bh:.6f}\n")
                
        return rotated, rotated_annotations
        
    def scale_image(self, image, scale):
        h, w = image.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    def eventFilter(self, source, event):
        # Handle keyboard events for file list navigation
        if source == self.file_list and event.type() == QEvent.KeyPress:
            key = event.key()
            
            if key == Qt.Key_Up:
                # Move to previous image
                current_row = self.file_list.currentRow()
                if current_row > 0:
                    # Only save if annotations were actually changed
                    if self.annotations_changed:
                        self.save_annotations()
                    self.file_list.setCurrentRow(current_row - 1)
                    self.current_image_index = current_row - 1
                    self.load_current_image()
                return True
                
            elif key == Qt.Key_Down:
                # Move to next image
                current_row = self.file_list.currentRow()
                if current_row < self.file_list.count() - 1:
                    # Only save if annotations were actually changed
                    if self.annotations_changed:
                        self.save_annotations()
                    self.file_list.setCurrentRow(current_row + 1)
                    self.current_image_index = current_row + 1
                    self.load_current_image()
                return True
                
        return super().eventFilter(source, event)
    
    def undo_action(self):
        self.image_canvas.undo()
        
    def redo_action(self):
        self.image_canvas.redo()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()