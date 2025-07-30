import sys
import os
import cv2
import numpy as np
import copy
import json
import random
import shutil
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QEvent
from PyQt5.QtGui import *
from ultralytics import SAM

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
        
        # For guide lines
        self.mouse_pos = None
        self.show_guides = True
        self.guide_color = (128, 128, 128)  # Gray color for guides
        self.setMouseTracking(True)  # Enable mouse tracking
        
        # For SAM2 mode
        self.sam2_mode = False
        self.sam2_box_mode = False
        self.sam_model = None
        self.sam_click_points = []
        self.sam_box_start = None
        self.sam_box_end = None
        self.sam_box_drawing = False
        # SAM2 parameters
        self.sam_conf_threshold = 0.5
        self.sam_stability_score = 0.95
        self.sam_iou_threshold = 0.88
        
    def set_sam2_mode(self, enabled):
        """Enable or disable SAM2 mode"""
        self.sam2_mode = enabled
        self.sam2_box_mode = False
        self.sam_click_points = []
        self.sam_box_start = None
        self.sam_box_end = None
        self.sam_box_drawing = False
        if enabled and self.sam_model is None:
            try:
                # Load SAM model
                self.sam_model = SAM("sam2.1_b.pt")
                print("SAM 2.1 model loaded successfully")
            except Exception as e:
                print(f"Failed to load SAM model: {e}")
                self.sam2_mode = False
                QMessageBox.warning(self, "SAM Model Error", f"Failed to load SAM model: {e}")
        self.update_display()
        
    def set_sam2_box_mode(self, enabled):
        """Enable or disable SAM2 Box mode"""
        self.sam2_box_mode = enabled
        self.sam2_mode = False
        self.sam_click_points = []
        self.sam_box_start = None
        self.sam_box_end = None
        self.sam_box_drawing = False
        if enabled and self.sam_model is None:
            try:
                # Load SAM model
                self.sam_model = SAM("sam2.1_b.pt")
                print("SAM 2.1 model loaded successfully for Box mode")
            except Exception as e:
                print(f"Failed to load SAM model: {e}")
                self.sam2_box_mode = False
                QMessageBox.warning(self, "SAM Model Error", f"Failed to load SAM model: {e}")
        self.update_display()
        
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
            
        # Draw SAM click points in SAM2 mode
        if self.sam2_mode and self.sam_click_points:
            for i, point in enumerate(self.sam_click_points):
                cv2.circle(temp_image, tuple(point), 5, (0, 0, 255), -1)
                cv2.putText(temp_image, str(i+1), 
                           (point[0]+10, point[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                           
        # Draw SAM box in SAM2 Box mode
        if self.sam2_box_mode:
            if self.sam_box_drawing and self.sam_box_start and self.sam_box_end:
                # Draw current box being drawn
                cv2.rectangle(temp_image, self.sam_box_start, self.sam_box_end, (255, 0, 255), 2)
                cv2.putText(temp_image, "SAM2 Box", 
                           (self.sam_box_start[0], self.sam_box_start[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
        # Draw guide lines
        if self.show_guides and self.mouse_pos and not self.dragging_bbox and not self.resizing_bbox:
            x, y = self.mouse_pos
            h, w = temp_image.shape[:2]
            
            # Vertical line
            cv2.line(temp_image, (x, 0), (x, h-1), self.guide_color, 1, cv2.LINE_AA)
            # Horizontal line
            cv2.line(temp_image, (0, y), (w-1, y), self.guide_color, 1, cv2.LINE_AA)
            
            # If drawing, show guides from start point
            if self.drawing and self.start_point:
                sx, sy = self.start_point
                # Vertical line from start point
                cv2.line(temp_image, (sx, 0), (sx, h-1), self.guide_color, 1, cv2.LINE_AA)
                # Horizontal line from start point
                cv2.line(temp_image, (0, sy), (w-1, sy), self.guide_color, 1, cv2.LINE_AA)
            
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
            if self.sam2_mode:
                # SAM2 mode - add click point and run segmentation
                self.sam_click_points.append(list(pos))
                self.update_display()
                
                # Run SAM segmentation
                self.run_sam_segmentation()
                
            elif self.sam2_box_mode:
                # SAM2 Box mode - start drawing box
                self.sam_box_drawing = True
                self.sam_box_start = pos
                self.sam_box_end = pos
                
            elif modifiers == Qt.ControlModifier:
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
            if self.sam2_mode and self.sam_click_points:
                # Remove last click point in SAM2 mode
                self.sam_click_points.pop()
                self.update_display()
            elif self.sam2_box_mode:
                # Cancel current box drawing in SAM2 Box mode
                self.sam_box_drawing = False
                self.sam_box_start = None
                self.sam_box_end = None
                self.update_display()
            else:
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
                    
    def run_sam_segmentation(self):
        """Run SAM segmentation with current click points"""
        if not self.sam_model or not self.sam_click_points:
            return
            
        try:
            # Run SAM inference with adjustable parameters
            if len(self.sam_click_points) == 1:
                results = self.sam_model.predict(
                    source=self.original_image,
                    points=self.sam_click_points[0],
                    conf=self.sam_conf_threshold,
                    iou=self.sam_iou_threshold
                )
            else:
                results = self.sam_model.predict(
                    source=self.original_image,
                    points=self.sam_click_points,
                    conf=self.sam_conf_threshold,
                    iou=self.sam_iou_threshold
                )
                
            if results and len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                
                # Get parent for class information
                parent = self.parent()
                while parent and not hasattr(parent, 'get_selected_class'):
                    parent = parent.parent()
                    
                if parent:
                    class_id, class_name = parent.get_selected_class()
                    self.selected_class_id = class_id
                    
                    # Filter masks by stability score and create bounding boxes
                    for i, mask in enumerate(masks):
                        # Apply mask threshold
                        mask_binary = mask > self.sam_conf_threshold
                        
                        # Calculate mask area for filtering small segments
                        mask_area = np.sum(mask_binary)
                        total_area = mask.shape[0] * mask.shape[1]
                        area_ratio = mask_area / total_area
                        
                        # Filter out very small segments (less than 0.01% of image)
                        if area_ratio < 0.0001:
                            continue
                            
                        # Find bounding box from mask
                        y_indices, x_indices = np.where(mask_binary)
                        if len(x_indices) > 0 and len(y_indices) > 0:
                            x1 = int(x_indices.min())
                            y1 = int(y_indices.min())
                            x2 = int(x_indices.max())
                            y2 = int(y_indices.max())
                            
                            # Create bounding box
                            bbox = BoundingBoxItem((x1, y1, x2, y2), class_id, class_name)
                            cmd = AddBBoxCommand(self, bbox)
                            cmd.execute()
                            self.execute_command(cmd)
                            
                            # For single point, only take the first (best) mask
                            if len(self.sam_click_points) == 1:
                                break
                            
                # Clear click points after successful segmentation
                self.sam_click_points = []
                self.update_display()
                        
        except Exception as e:
            print(f"SAM segmentation error: {e}")
            # Fallback to original method if predict fails
            try:
                if len(self.sam_click_points) == 1:
                    results = self.sam_model(self.original_image, points=self.sam_click_points[0])
                else:
                    results = self.sam_model(self.original_image, points=self.sam_click_points)
                    
                if results and len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
                    masks = results[0].masks.data.cpu().numpy()
                    if len(masks) > 0:
                        mask = masks[0]
                        
                        # Find bounding box from mask
                        y_indices, x_indices = np.where(mask > self.sam_conf_threshold)
                        if len(x_indices) > 0 and len(y_indices) > 0:
                            x1 = int(x_indices.min())
                            y1 = int(y_indices.min())
                            x2 = int(x_indices.max())
                            y2 = int(y_indices.max())
                            
                            # Add bounding box
                            parent = self.parent()
                            while parent and not hasattr(parent, 'get_selected_class'):
                                parent = parent.parent()
                                
                            if parent:
                                class_id, class_name = parent.get_selected_class()
                                self.selected_class_id = class_id
                                bbox = BoundingBoxItem((x1, y1, x2, y2), class_id, class_name)
                                cmd = AddBBoxCommand(self, bbox)
                                cmd.execute()
                                self.execute_command(cmd)
                                
                            # Clear click points after successful segmentation
                            self.sam_click_points = []
                            self.update_display()
            except Exception as e2:
                print(f"SAM fallback segmentation error: {e2}")
                
    def run_sam_box_segmentation(self, box_rect):
        """Run SAM segmentation with bounding box prompt"""
        if not self.sam_model:
            return
            
        try:
            # Convert box format for SAM: [x1, y1, x2, y2]
            box_prompt = box_rect
            
            # Run SAM inference with box prompt
            try:
                results = self.sam_model.predict(
                    source=self.original_image,
                    bboxes=box_prompt,
                    conf=self.sam_conf_threshold,
                    iou=self.sam_iou_threshold
                )
            except:
                # Fallback to original method if predict fails
                results = self.sam_model(self.original_image, bboxes=box_prompt)
                
            if results and len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                
                # Get parent for class information
                parent = self.parent()
                while parent and not hasattr(parent, 'get_selected_class'):
                    parent = parent.parent()
                    
                if parent and len(masks) > 0:
                    class_id, class_name = parent.get_selected_class()
                    self.selected_class_id = class_id
                    
                    # Use the best mask (first one)
                    mask = masks[0]
                    
                    # Apply mask threshold
                    mask_binary = mask > self.sam_conf_threshold
                    
                    # Find refined bounding box from mask
                    y_indices, x_indices = np.where(mask_binary)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x1 = int(x_indices.min())
                        y1 = int(y_indices.min())
                        x2 = int(x_indices.max())
                        y2 = int(y_indices.max())
                        
                        # Create refined bounding box
                        bbox = BoundingBoxItem((x1, y1, x2, y2), class_id, class_name)
                        cmd = AddBBoxCommand(self, bbox)
                        cmd.execute()
                        self.execute_command(cmd)
                        
                        print(f"SAM2 Box: Refined bbox from {box_rect} to ({x1}, {y1}, {x2}, {y2})")
                        
                self.update_display()
                        
        except Exception as e:
            print(f"SAM box segmentation error: {e}")
            # Fallback: create bbox from original box
            parent = self.parent()
            while parent and not hasattr(parent, 'get_selected_class'):
                parent = parent.parent()
                
            if parent:
                class_id, class_name = parent.get_selected_class()
                self.selected_class_id = class_id
                bbox = BoundingBoxItem(tuple(box_rect), class_id, class_name)
                cmd = AddBBoxCommand(self, bbox)
                cmd.execute()
                self.execute_command(cmd)
                self.update_display()
                
    def set_sam_parameters(self, conf_threshold, stability_score, iou_threshold):
        """Set SAM2 sensitivity parameters"""
        self.sam_conf_threshold = conf_threshold
        self.sam_stability_score = stability_score
        self.sam_iou_threshold = iou_threshold
            
    def mouseMoveEvent(self, event):
        if self.original_image is None:
            return
            
        pos = self.get_image_coordinates(event.pos())
        if pos is None:
            self.mouse_pos = None
            self.update_display()
            return
            
        # Update mouse position for guide lines
        self.mouse_pos = pos
        
        if self.drawing:
            self.end_point = pos
            self.update_display()
        elif self.sam_box_drawing:
            # Update SAM2 box end point
            self.sam_box_end = pos
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
            # Update display to show guide lines
            self.update_display()
                
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.sam_box_drawing:
                # Finish SAM2 box drawing and run segmentation
                self.sam_box_drawing = False
                if self.sam_box_start and self.sam_box_end:
                    # Normalize coordinates
                    x1, y1 = self.sam_box_start
                    x2, y2 = self.sam_box_end
                    box_rect = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    
                    # Check minimum size (10 pixels)
                    width = box_rect[2] - box_rect[0]
                    height = box_rect[3] - box_rect[1]
                    
                    if width >= 10 and height >= 10:
                        # Run SAM with bbox prompt
                        self.run_sam_box_segmentation(box_rect)
                        
                    # Clear box
                    self.sam_box_start = None
                    self.sam_box_end = None
                    
                self.update_display()
                
            elif self.drawing and not self.sam2_mode and not self.sam2_box_mode:
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
        self.mouse_pos = None  # Clear mouse position
        self.update_display()  # Remove guide lines
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
        if self.sam2_mode:
            self.setCursor(Qt.PointingHandCursor)
        elif self.sam2_box_mode:
            self.setCursor(Qt.CrossCursor)
        else:
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

class SAM2SettingsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("SAM2 감도 설정")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Description
        desc = QLabel("SAM2 모델의 세그멘테이션 감도를 조절합니다.\n값이 낮을수록 더 많은 영역을 포함합니다.")
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("padding: 10px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Confidence threshold
        conf_layout = QVBoxLayout()
        conf_label = QLabel("Confidence Threshold (객체 인식 임계값)")
        conf_layout.addWidget(conf_label)
        
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)  # 0.1 to 0.9
        self.conf_slider.setValue(50)  # 0.5
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        conf_layout.addWidget(self.conf_slider)
        
        self.conf_value_label = QLabel("0.50")
        self.conf_value_label.setAlignment(Qt.AlignCenter)
        conf_layout.addWidget(self.conf_value_label)
        
        layout.addLayout(conf_layout)
        
        # IoU threshold
        iou_layout = QVBoxLayout()
        iou_label = QLabel("IoU Threshold (영역 겹침 임계값)")
        iou_layout.addWidget(iou_label)
        
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(50, 95)  # 0.5 to 0.95
        self.iou_slider.setValue(88)  # 0.88
        self.iou_slider.valueChanged.connect(self.update_iou_label)
        iou_layout.addWidget(self.iou_slider)
        
        self.iou_value_label = QLabel("0.88")
        self.iou_value_label.setAlignment(Qt.AlignCenter)
        iou_layout.addWidget(self.iou_value_label)
        
        layout.addLayout(iou_layout)
        
        # Stability score threshold
        stability_layout = QVBoxLayout()
        stability_label = QLabel("Stability Score (분할 안정성)")
        stability_layout.addWidget(stability_label)
        
        self.stability_slider = QSlider(Qt.Horizontal)
        self.stability_slider.setRange(80, 99)  # 0.8 to 0.99
        self.stability_slider.setValue(95)  # 0.95
        self.stability_slider.valueChanged.connect(self.update_stability_label)
        stability_layout.addWidget(self.stability_slider)
        
        self.stability_value_label = QLabel("0.95")
        self.stability_value_label.setAlignment(Qt.AlignCenter)
        stability_layout.addWidget(self.stability_value_label)
        
        layout.addLayout(stability_layout)
        
        # Presets
        preset_layout = QVBoxLayout()
        preset_label = QLabel("프리셋")
        preset_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        preset_layout.addWidget(preset_label)
        
        preset_buttons_layout = QHBoxLayout()
        
        self.high_sensitivity_btn = QPushButton("고감도\n(더 많은 영역)")
        self.high_sensitivity_btn.clicked.connect(self.set_high_sensitivity)
        preset_buttons_layout.addWidget(self.high_sensitivity_btn)
        
        self.normal_sensitivity_btn = QPushButton("보통\n(기본값)")
        self.normal_sensitivity_btn.clicked.connect(self.set_normal_sensitivity)
        preset_buttons_layout.addWidget(self.normal_sensitivity_btn)
        
        self.low_sensitivity_btn = QPushButton("저감도\n(정확한 영역)")
        self.low_sensitivity_btn.clicked.connect(self.set_low_sensitivity)
        preset_buttons_layout.addWidget(self.low_sensitivity_btn)
        
        preset_layout.addLayout(preset_buttons_layout)
        layout.addLayout(preset_layout)
        
        # Apply button
        self.apply_button = QPushButton("설정 적용")
        self.apply_button.setStyleSheet("padding: 10px; font-size: 14px; margin-top: 10px;")
        layout.addWidget(self.apply_button)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def update_conf_label(self, value):
        self.conf_value_label.setText(f"{value/100:.2f}")
        
    def update_iou_label(self, value):
        self.iou_value_label.setText(f"{value/100:.2f}")
        
    def update_stability_label(self, value):
        self.stability_value_label.setText(f"{value/100:.2f}")
        
    def set_high_sensitivity(self):
        """고감도 설정 - 더 많은 영역을 포함"""
        self.conf_slider.setValue(30)  # 0.3
        self.iou_slider.setValue(70)   # 0.7
        self.stability_slider.setValue(85)  # 0.85
        
    def set_normal_sensitivity(self):
        """보통 감도 설정 - 기본값"""
        self.conf_slider.setValue(50)  # 0.5
        self.iou_slider.setValue(88)   # 0.88
        self.stability_slider.setValue(95)  # 0.95
        
    def set_low_sensitivity(self):
        """저감도 설정 - 정확한 영역만"""
        self.conf_slider.setValue(70)  # 0.7
        self.iou_slider.setValue(92)   # 0.92
        self.stability_slider.setValue(98)  # 0.98
        
    def get_parameters(self):
        """현재 설정된 파라미터 반환"""
        return {
            'conf_threshold': self.conf_slider.value() / 100.0,
            'iou_threshold': self.iou_slider.value() / 100.0,
            'stability_score': self.stability_slider.value() / 100.0
        }

class DataSplitWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("데이터 분할 설정")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Description
        desc = QLabel("데이터셋을 train, valid, test로 분할합니다.\n비율의 합은 1.0이어야 합니다.")
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("padding: 10px;")
        layout.addWidget(desc)
        
        # Ratio inputs
        form_layout = QFormLayout()
        
        # Train ratio
        self.train_ratio = QDoubleSpinBox()
        self.train_ratio.setRange(0.0, 1.0)
        self.train_ratio.setSingleStep(0.1)
        self.train_ratio.setValue(0.7)
        self.train_ratio.valueChanged.connect(self.check_ratio_sum)
        form_layout.addRow("Train ratio:", self.train_ratio)
        
        # Valid ratio
        self.valid_ratio = QDoubleSpinBox()
        self.valid_ratio.setRange(0.0, 1.0)
        self.valid_ratio.setSingleStep(0.1)
        self.valid_ratio.setValue(0.2)
        self.valid_ratio.valueChanged.connect(self.check_ratio_sum)
        form_layout.addRow("Valid ratio:", self.valid_ratio)
        
        # Test ratio
        self.test_ratio = QDoubleSpinBox()
        self.test_ratio.setRange(0.0, 1.0)
        self.test_ratio.setSingleStep(0.1)
        self.test_ratio.setValue(0.1)
        self.test_ratio.valueChanged.connect(self.check_ratio_sum)
        form_layout.addRow("Test ratio:", self.test_ratio)
        
        layout.addLayout(form_layout)
        
        # Sum display
        self.sum_label = QLabel("합계: 1.0")
        self.sum_label.setAlignment(Qt.AlignCenter)
        self.sum_label.setStyleSheet("padding: 10px; font-weight: bold;")
        layout.addWidget(self.sum_label)
        
        # Warning label
        self.warning_label = QLabel("")
        self.warning_label.setAlignment(Qt.AlignCenter)
        self.warning_label.setStyleSheet("color: red; padding: 5px;")
        layout.addWidget(self.warning_label)
        
        # Working directory display
        self.working_dir_label = QLabel("Working Dir: 없음")
        self.working_dir_label.setStyleSheet("padding: 10px;")
        self.working_dir_label.setWordWrap(True)
        layout.addWidget(self.working_dir_label)
        
        # Select working directory button
        self.select_working_dir_btn = QPushButton("Working Dir 선택")
        self.select_working_dir_btn.clicked.connect(self.select_working_dir)
        layout.addWidget(self.select_working_dir_btn)
        
        # Apply button
        self.apply_button = QPushButton("데이터 분할 실행")
        self.apply_button.setStyleSheet("padding: 10px; font-size: 14px;")
        layout.addWidget(self.apply_button)
        
        layout.addStretch()
        self.setLayout(layout)
        
        self.working_dir = ""
        
    def check_ratio_sum(self):
        total = self.train_ratio.value() + self.valid_ratio.value() + self.test_ratio.value()
        self.sum_label.setText(f"합계: {total:.1f}")
        
        if abs(total - 1.0) > 0.001:
            self.warning_label.setText("비율의 합이 1.0이 아닙니다!")
            self.apply_button.setEnabled(False)
        else:
            self.warning_label.setText("")
            self.apply_button.setEnabled(True)
            
    def select_working_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Working Directory")
        if directory:
            self.working_dir = directory
            self.working_dir_label.setText(f"Working Dir: {os.path.basename(directory)}")
            self.working_dir_label.setToolTip(directory)

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
        
        # Add Data Split button
        self.data_split_btn = QPushButton("Data Split")
        self.data_split_btn.setCheckable(True)
        self.data_split_btn.clicked.connect(self.toggle_data_split)
        left_menu.addWidget(self.data_split_btn)
        
        # Add SAM2 button
        self.sam2_btn = QPushButton("SAM2")
        self.sam2_btn.setCheckable(True)
        self.sam2_btn.clicked.connect(self.toggle_sam2_mode)
        self.sam2_btn.setStyleSheet("QPushButton:checked { background-color: #4CAF50; }")
        left_menu.addWidget(self.sam2_btn)
        
        # Add SAM2 Box button
        self.sam2_box_btn = QPushButton("SAM2 Box")
        self.sam2_box_btn.setCheckable(True)
        self.sam2_box_btn.clicked.connect(self.toggle_sam2_box_mode)
        self.sam2_box_btn.setStyleSheet("QPushButton:checked { background-color: #2196F3; }")
        left_menu.addWidget(self.sam2_box_btn)
        
        # Add SAM2 Settings button
        self.sam2_settings_btn = QPushButton("SAM2 Settings")
        self.sam2_settings_btn.setCheckable(True)
        self.sam2_settings_btn.clicked.connect(self.toggle_sam2_settings)
        left_menu.addWidget(self.sam2_settings_btn)
        
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
        
        # Data split widget
        self.split_widget = DataSplitWidget()
        self.split_widget.apply_button.clicked.connect(self.apply_data_split)
        self.stacked_widget.addWidget(self.split_widget)
        
        # SAM2 settings widget
        self.sam2_settings_widget = SAM2SettingsWidget()
        self.sam2_settings_widget.apply_button.clicked.connect(self.apply_sam2_settings)
        self.stacked_widget.addWidget(self.sam2_settings_widget)
        
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
            # Uncheck other toggle buttons
            self.data_split_btn.setChecked(False)
            self.sam2_btn.setChecked(False)
            self.sam2_box_btn.setChecked(False)
            self.sam2_settings_btn.setChecked(False)
            self.image_canvas.set_sam2_mode(False)
            self.image_canvas.set_sam2_box_mode(False)
        else:
            self.stacked_widget.setCurrentIndex(0)
            self.data_aug_btn.setText("Data Aug")
            
    def toggle_data_split(self, checked):
        if checked:
            self.stacked_widget.setCurrentIndex(2)
            self.data_split_btn.setText("Show Image")
            # Uncheck other toggle buttons
            self.data_aug_btn.setChecked(False)
            self.sam2_btn.setChecked(False)
            self.sam2_box_btn.setChecked(False)
            self.sam2_settings_btn.setChecked(False)
            self.image_canvas.set_sam2_mode(False)
            self.image_canvas.set_sam2_box_mode(False)
        else:
            self.stacked_widget.setCurrentIndex(0)
            self.data_split_btn.setText("Data Split")
            
    def toggle_sam2_mode(self, checked):
        """Toggle SAM2 mode on/off"""
        self.image_canvas.set_sam2_mode(checked)
        if checked:
            # Return to image view
            self.stacked_widget.setCurrentIndex(0)
            # Uncheck other toggle buttons
            self.data_aug_btn.setChecked(False)
            self.data_split_btn.setChecked(False)
            self.sam2_box_btn.setChecked(False)
            self.sam2_settings_btn.setChecked(False)
            # Disable other SAM modes
            self.image_canvas.set_sam2_box_mode(False)
            # Update button text
            self.sam2_btn.setText("SAM2 ON")
            # Show info message
            QMessageBox.information(self, "SAM2 Mode", 
                                  "SAM2 모드가 활성화되었습니다.\n\n"
                                  "사용법:\n"
                                  "- 왼쪽 클릭: 객체 위에 포인트 추가\n"
                                  "- 오른쪽 클릭: 마지막 포인트 제거\n"
                                  "- 포인트를 클릭하면 자동으로 분할됩니다.\n\n"
                                  "SAM2 Settings에서 감도를 조절할 수 있습니다.")
        else:
            self.sam2_btn.setText("SAM2")
            
    def toggle_sam2_box_mode(self, checked):
        """Toggle SAM2 Box mode on/off"""
        self.image_canvas.set_sam2_box_mode(checked)
        if checked:
            # Return to image view
            self.stacked_widget.setCurrentIndex(0)
            # Uncheck other toggle buttons
            self.data_aug_btn.setChecked(False)
            self.data_split_btn.setChecked(False)
            self.sam2_btn.setChecked(False)
            self.sam2_settings_btn.setChecked(False)
            # Disable other SAM modes
            self.image_canvas.set_sam2_mode(False)
            # Update button text
            self.sam2_box_btn.setText("SAM2 Box ON")
            # Show info message
            QMessageBox.information(self, "SAM2 Box Mode", 
                                  "SAM2 Box 모드가 활성화되었습니다.\n\n"
                                  "사용법:\n"
                                  "- 마우스 드래그: 객체보다 조금 더 큰 영역을 박스로 그리기\n"
                                  "- 오른쪽 클릭: 현재 박스 취소\n"
                                  "- 박스를 완성하면 SAM2가 정밀한 바운딩 박스를 생성합니다.\n\n"
                                  "SAM2 Settings에서 감도를 조절할 수 있습니다.")
        else:
            self.sam2_box_btn.setText("SAM2 Box")
            
    def toggle_sam2_settings(self, checked):
        """Toggle SAM2 settings view"""
        if checked:
            self.stacked_widget.setCurrentIndex(3)
            self.sam2_settings_btn.setText("Show Image")
            # Uncheck other toggle buttons
            self.data_aug_btn.setChecked(False)
            self.data_split_btn.setChecked(False)
            self.sam2_btn.setChecked(False)
            self.sam2_box_btn.setChecked(False)
            self.image_canvas.set_sam2_mode(False)
            self.image_canvas.set_sam2_box_mode(False)
        else:
            self.stacked_widget.setCurrentIndex(0)
            self.sam2_settings_btn.setText("SAM2 Settings")
            
    def apply_sam2_settings(self):
        """Apply SAM2 sensitivity settings"""
        params = self.sam2_settings_widget.get_parameters()
        self.image_canvas.set_sam_parameters(
            params['conf_threshold'],
            params['stability_score'],
            params['iou_threshold']
        )
        QMessageBox.information(self, "설정 적용", 
                              f"SAM2 설정이 적용되었습니다.\n\n"
                              f"Confidence: {params['conf_threshold']:.2f}\n"
                              f"IoU: {params['iou_threshold']:.2f}\n"
                              f"Stability: {params['stability_score']:.2f}")
            
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
    
    def apply_data_split(self):
        if not self.current_directory or not self.label_directory:
            QMessageBox.warning(self, "경고", "먼저 Image Dir과 Label Dir을 선택하세요.")
            return
            
        if not self.split_widget.working_dir:
            QMessageBox.warning(self, "경고", "Working Dir을 선택하세요.")
            return
            
        train_ratio = self.split_widget.train_ratio.value()
        valid_ratio = self.split_widget.valid_ratio.value()
        test_ratio = self.split_widget.test_ratio.value()
        
        # Check ratio sum
        if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 0.001:
            QMessageBox.warning(self, "경고", "비율의 합이 1.0이 아닙니다.")
            return
            
        # Create data directory structure
        data_dir = os.path.join(self.split_widget.working_dir, "data")
        train_dir = os.path.join(data_dir, "train")
        valid_dir = os.path.join(data_dir, "valid")
        test_dir = os.path.join(data_dir, "test")
        
        # Create directories
        for split_dir in [train_dir, valid_dir, test_dir]:
            os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(split_dir, "labels"), exist_ok=True)
            
        # Get all image files and their corresponding labels
        image_label_pairs = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for file in os.listdir(self.current_directory):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(self.current_directory, file)
                label_file = os.path.splitext(file)[0] + '.txt'
                label_path = os.path.join(self.label_directory, label_file)
                
                # Only include if label file exists
                if os.path.exists(label_path):
                    image_label_pairs.append((image_path, label_path, file, label_file))
                    
        if not image_label_pairs:
            QMessageBox.warning(self, "경고", "라벨이 있는 이미지 파일이 없습니다.")
            return
            
        # Shuffle for random split
        random.shuffle(image_label_pairs)
        
        # Calculate split indices
        total_files = len(image_label_pairs)
        train_count = int(total_files * train_ratio)
        valid_count = int(total_files * valid_ratio)
        
        # Split files
        train_files = image_label_pairs[:train_count]
        valid_files = image_label_pairs[train_count:train_count + valid_count]
        test_files = image_label_pairs[train_count + valid_count:]
        
        # Progress dialog
        progress = QProgressDialog("데이터 분할 중...", "취소", 0, total_files, self)
        progress.setWindowModality(Qt.WindowModal)
        
        # Copy files to respective directories
        file_count = 0
        for split_name, split_files, split_dir in [
            ("train", train_files, train_dir),
            ("valid", valid_files, valid_dir),
            ("test", test_files, test_dir)
        ]:
            for image_path, label_path, image_name, label_name in split_files:
                if progress.wasCanceled():
                    return
                    
                progress.setValue(file_count)
                
                # Copy image
                dest_image = os.path.join(split_dir, "images", image_name)
                shutil.copy2(image_path, dest_image)
                
                # Copy label
                dest_label = os.path.join(split_dir, "labels", label_name)
                shutil.copy2(label_path, dest_label)
                
                file_count += 1
                
        progress.setValue(total_files)
        
        QMessageBox.information(self, "완료", 
                              f"데이터 분할이 완료되었습니다.\n\n"
                              f"전체: {total_files}개 파일\n"
                              f"• Train: {len(train_files)}개\n"
                              f"• Valid: {len(valid_files)}개\n"
                              f"• Test: {len(test_files)}개\n\n"
                              f"저장 위치: {data_dir}")
    
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