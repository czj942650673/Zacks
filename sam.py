import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import json
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

np.random.seed(3)

class SAMInteractiveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("阿君蜜汁小工具-SAM")
        self.root.geometry("1000x800")
        
        # 初始化变量
        self.image = None
        self.image_path = None
        self.model = None
        self.predictor = None
        self.points = []
        self.labels = []
        self.masks = None
        self.scores = None
        self.current_point_index = 0
        self.display_img = None  # 存储显示的图像引用
        self.img_scale = 1.0  # 图像缩放比例
        self.img_offset = (0, 0)  # 图像偏移量
        self.image_results = {}  # 存储每张图片的分割结果缓存
        self.current_image_path = None  # 当前图片路径
        
        # 新增目录导航变量
        self.image_dir = None
        self.image_list = []
        self.current_image_index = -1
        
        # 多类分割相关变量
        self.class_names = []  # 存储所有已定义的类别名称
        self.class_to_id = {}  # 类别名称到ID的映射
        self.image_annotations = {}  # 存储每张图片的标注信息，键为图片路径
        self.current_mask_id = 0  # 当前掩码ID，用于显示不同颜色
        self.current_image_annotations = []  # 当前图片的所有标注信息
        self.current_mask_bbox = None  # 当前掩码的最小矩形框
        
        # 鼠标交互相关变量
        self.selected_point_index = -1  # 选中的锚点索引
        self.hovered_mask_index = -1  # 当前悬停的掩码索引
        
        # 创建UI
        self._create_widgets()
        
        # 绑定键盘事件
        self.root.bind_all('<KeyPress-a>', lambda event: self.show_previous_image())
        self.root.bind_all('<KeyPress-d>', lambda event: self.show_next_image())
        # 绑定快捷键
        self.root.bind_all('<KeyPress-x>', lambda event: self.perform_segmentation())
        self.root.bind_all('<Control-s>', lambda event: self.save_all_results())
        # 使用bind_all确保在任何组件获得焦点时都能响应
        self.root.bind_all('<KeyPress-s>', lambda event: self.save_current_object())
        # 绑定重置快捷键
        self.root.bind_all('<KeyPress-r>', lambda event: self.reset_app())

    
    def _create_widgets(self):
        # 创建顶部控制区域
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # 加载图片按钮
        self.load_path_btn = ttk.Button(control_frame, text="加载图片/路径", command=self.load_image_or_directory)
        self.load_path_btn.pack(side=tk.LEFT, padx=5)
        
        # 加载模型按钮
        self.load_model_btn = ttk.Button(control_frame, text="加载模型", command=self.load_model)
        self.load_model_btn.pack(side=tk.LEFT, padx=5)
        
        # 添加上一张/下一张按钮
        self.prev_btn = ttk.Button(control_frame, text="上一张(A)", command=self.show_previous_image, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = ttk.Button(control_frame, text="下一张(D)", command=self.show_next_image, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        # 自动累加锚点 - 无需设置锚点数量
        
        # 分割按钮
        self.segment_btn = ttk.Button(control_frame, text="执行分割(X)", command=self.perform_segmentation, state=tk.DISABLED)
        self.segment_btn.pack(side=tk.LEFT, padx=5)
        
        # 保存当前目标按钮
        self.save_object_btn = ttk.Button(control_frame, text="保存类别(S)", 
                                        command=self.save_current_object, state=tk.DISABLED)
        self.save_object_btn.pack(side=tk.LEFT, padx=5)

        # 保存所有标签按钮
        self.save_all_btn = ttk.Button(control_frame, text="保存所有标签(Ctrl+S)", 
                                  command=self.save_all_results, state=tk.DISABLED)
        self.save_all_btn.pack(side=tk.LEFT, padx=5)
        

        
        # 数据集类型选择下拉框
        tk.Label(control_frame, text="数据集类型：").pack(side=tk.LEFT, padx=5)
        self.dataset_type = tk.StringVar(value="分割")
        self.dataset_type_combo = ttk.Combobox(control_frame, textvariable=self.dataset_type, state="readonly", width=10)
        self.dataset_type_combo['values'] = ("分割", "检测")
        self.dataset_type_combo.pack(side=tk.LEFT, padx=5)

        # 加载标签按钮
        self.load_labels_btn = ttk.Button(control_frame, text="加载标签", command=self.load_labels, state=tk.DISABLED)
        self.load_labels_btn.pack(side=tk.LEFT, padx=5)
        
        # 重置按钮
        self.reset_btn = ttk.Button(control_frame, text="重置(R)", command=self.reset_app)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # 创建状态标签
        self.status_var = tk.StringVar(value="请加载图片和模型")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 创建图片显示区域
        self.canvas_frame = ttk.Frame(self.root, padding="10")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        # 绑定Ctrl+滚轮缩放事件
        self.canvas.bind("<Control-MouseWheel>", self.on_zoom)
        # 用于跟踪Ctrl键状态
        self.ctrl_pressed = False
        self.root.bind("<Control-KeyPress>", lambda e: setattr(self, "ctrl_pressed", True))
        self.root.bind("<Control-KeyRelease>", lambda e: setattr(self, "ctrl_pressed", False))
        
        # 当前选中的锚点索引
        self.selected_point_index = -1
        
        # 创建点列表显示区域
        self.points_frame = ttk.LabelFrame(self.root, text="已选点", padding="10")
        self.points_frame.pack(fill=tk.BOTH, expand=False, side=tk.BOTTOM)
        
        self.points_text = tk.Text(self.points_frame, height=5, width=50)
        self.points_text.pack(fill=tk.BOTH, expand=True)
    
    def load_image(self, file_path=None):
        if file_path is None:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
            )
        
        if file_path:
            # 保存当前图片的结果到缓存
            if self.current_image_path and self.masks is not None:
                self.image_results[self.current_image_path] = {
                    'masks': self.masks,
                    'scores': self.scores,
                    'points': self.points,
                    'labels': self.labels
                }
            
            # 保存当前图片的标注信息
            if self.current_image_path and self.current_image_annotations:
                # 使用save_current_object方法中的相同逻辑，确保不会重复保存标注
                if self.current_image_path not in self.image_annotations:
                    self.image_annotations[self.current_image_path] = []
                # 只保存有变化的标注，避免重复
                if self.current_image_path not in self.image_annotations or len(self.current_image_annotations) != len(self.image_annotations[self.current_image_path]):
                    self.image_annotations[self.current_image_path] = self.current_image_annotations.copy()
            
            self.image_path = file_path
            self.current_image_path = file_path
            
            # 加载原始图片
            img = Image.open(file_path).convert("RGB")
            self.image = np.array(img)
            
            # 检查是否有缓存结果
            if file_path in self.image_results:
                cached = self.image_results[file_path]
                self.masks = cached['masks']
                self.scores = cached['scores']
                self.points = cached['points']
                self.labels = cached['labels']
                self._update_points_display()
            else:
                self.masks = None
                self.scores = None
                self.points = []
                self.labels = []
                self.current_point_index = 0
                self.points_text.delete(1.0, tk.END)
            
            # 清除当前掩码的最小矩形框，确保它不会显示在新图像上
            self.current_mask_bbox = None
            
            # 检查是否有缓存的标注信息
            if file_path in self.image_annotations:
                self.current_image_annotations = self.image_annotations[file_path]
            else:
                self.current_image_annotations = []
            
            # 确保UI布局已更新以获取正确的画布尺寸
            self.root.update_idletasks()
            self.canvas.update_idletasks()
            self._display_image()
            self.status_var.set(f"已加载图片: {os.path.basename(file_path)}")
            self._check_enable_segment()
            self._check_enable_save_object()
            self._check_enable_save_buttons()
    
    def load_image_or_directory(self):
        # 询问用户选择文件还是目录
        # 创建自定义对话框选择加载类型
        dialog = tk.Toplevel(self.root)
        dialog.title("选择加载类型")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="请选择加载类型:", pady=10).pack()
        
        frame = ttk.Frame(dialog)
        frame.pack(pady=10)
        
        def choose_image():
            dialog.destroy()
            file_path = filedialog.askopenfilename(
                title="选择图片文件",
                initialdir=os.getcwd(),
                filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All Files", "*")]
            )
            if file_path:
                self.load_image(file_path)
        
        def choose_directory():
            dialog.destroy()
            dir_path = filedialog.askdirectory(
                title="选择图片目录",
                initialdir=os.getcwd()
            )
            if dir_path:
                self.load_image_directory(dir_path)
        
        ttk.Button(frame, text="加载图片", command=choose_image).pack(side=tk.LEFT, padx=10)
        ttk.Button(frame, text="加载路径", command=choose_directory).pack(side=tk.LEFT, padx=10)

    def load_image_directory(self, dir_path=None):
        if dir_path is None:
            dir_path = filedialog.askdirectory()
        if dir_path:
            self.image_dir = dir_path
            self.image_list = []
            
            # 支持的图片扩展名
            img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            
            # 获取目录中所有图片文件
            for filename in sorted(os.listdir(dir_path)):
                ext = os.path.splitext(filename)[1].lower()
                if ext in img_extensions:
                    self.image_list.append(os.path.join(dir_path, filename))
            
            if self.image_list:
                self.current_image_index = 0
                self.load_image(self.image_list[self.current_image_index])
                self.status_var.set(f"已加载目录: {dir_path} (共{len(self.image_list)}张图片)")
                self.prev_btn.config(state=tk.NORMAL)
                self.next_btn.config(state=tk.NORMAL)
            else:
                messagebox.showinfo("信息", "目录中没有找到图片文件")
    
    def show_previous_image(self):
        if self.image_list and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image(self.image_list[self.current_image_index])
            self.status_var.set(f"图片 {self.current_image_index + 1}/{len(self.image_list)}: {os.path.basename(self.image_path)}")
    
    def show_next_image(self):
        if self.image_list and self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.load_image(self.image_list[self.current_image_index])
            self.status_var.set(f"图片 {self.current_image_index + 1}/{len(self.image_list)}: {os.path.basename(self.image_path)}")
    
    def load_model(self):
        checkpoint_path = filedialog.askopenfilename(
            title="选择模型文件",
            initialdir=os.getcwd(),
            filetypes=[("Model files", "*.pt")]
        )
        if checkpoint_path:
            try:
                # 获取项目根目录
                project_root = os.path.dirname(os.path.abspath(__file__))
                
                # 确定模型类型并选择正确的配置文件
                checkpoint_name = os.path.basename(checkpoint_path)
                if "large" in checkpoint_name:
                    config_filename = "sam2_hiera_l.yaml"
                elif "base" in checkpoint_name:
                    config_filename = "sam2_hiera_b+.yaml"
                elif "small" in checkpoint_name:
                    config_filename = "sam2_hiera_s.yaml"
                elif "tiny" in checkpoint_name:
                    config_filename = "sam2_hiera_t.yaml"
                else:
                    # 默认使用large配置
                    config_filename = "sam2_hiera_l.yaml"
                    messagebox.showinfo("信息", "未识别的模型类型，使用默认配置")
                
                # 构建配置文件的可能路径
                config_path = os.path.join(
                    project_root, "sam2", "configs", "sam2", config_filename
                )
                
                # 如果第一个路径不存在，尝试直接在sam2目录下查找
                if not os.path.exists(config_path):
                    config_path = os.path.join(
                        project_root, "sam2", config_filename
                    )
                
                # 验证配置文件是否存在
                if not os.path.exists(config_path):
                    messagebox.showerror("错误", f"配置文件不存在: {config_path}")
                    return
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = build_sam2(config_path, checkpoint_path, device=device)
                self.predictor = SAM2ImagePredictor(self.model)
                self.status_var.set(f"已加载模型: {os.path.basename(checkpoint_path)} ({device})")
                self._check_enable_segment()
            except Exception as e:
                messagebox.showerror("错误", f"加载模型失败: {str(e)}")
                self.status_var.set("加载模型失败")
    
    def apply_num_points(self):
        try:
            self.reset_points()
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")
    
    def reset_app(self):
        # 清除锚点、分割结果和缓存
        self.points = []
        self.labels = []
        self.masks = None
        self.scores = None
        self.current_point_index = 0
        self.current_mask_bbox = None  # 清除最小矩阵框
        
        # 清除当前图片的缓存结果
        if self.current_image_path and self.current_image_path in self.image_results:
            del self.image_results[self.current_image_path]
        
        self.canvas.delete("all")
        self.points_text.delete(1.0, tk.END)
        self.status_var.set("已重置锚点和分割结果")
        self._display_image()  # 重新显示图片

        self.save_object_btn.config(state=tk.DISABLED)

    def on_canvas_click(self, event):
        if self.image is None:
            return

        # 获取画布上的点击位置
        x, y = event.x, event.y

        # 计算实际图像坐标
        img_x = int((x - self.img_offset[0]) / self.img_scale)
        img_y = int((y - self.img_offset[1]) / self.img_scale)

        # 确保坐标在图像范围内
        img_height, img_width = self.image.shape[:2]
        img_x = min(max(img_x, 0), img_width - 1)
        img_y = min(max(img_y, 0), img_height - 1)

        # 添加点和标签
        self.points.append([img_x, img_y])
        self.labels.append(1)  # 默认是正点

        # 更新显示
        self._update_points_display()
        self._display_image()

        # 自动累加锚点
        self.status_var.set(f"已添加锚点 {len(self.points)}")
        self._check_enable_segment()
        
    def _check_enable_save_object(self):
        """检查是否可以启用保存目标按钮"""
        if self.image is not None and self.masks is not None:
            self.save_object_btn.config(state=tk.NORMAL)
        else:
            self.save_object_btn.config(state=tk.DISABLED)
            
    def _check_enable_save_buttons(self):
        """检查是否可以启用保存按钮"""
        # 检查是否有任何图像有标注结果
        has_annotations = False
        # 添加调试信息
        print(f"当前image_annotations字典内容: {self.image_annotations}")
        print(f"当前current_image_path: {self.current_image_path}")
        
        # 先检查当前图像是否有标注
        if self.current_image_path and self.current_image_path in self.image_annotations:
            current_annotations = self.image_annotations[self.current_image_path]
            if current_annotations:
                has_annotations = True
                print(f"当前图像 {self.current_image_path} 有 {len(current_annotations)} 个标注")
        
        # 如果当前图像没有标注，再遍历整个字典检查其他图像
        if not has_annotations:
            for path, annotations in self.image_annotations.items():
                print(f"路径 {path} 有 {len(annotations)} 个标注")
                if annotations:
                    has_annotations = True
                    break
        
        if has_annotations:
            self.save_all_btn.config(state=tk.NORMAL)
        else:
            self.save_all_btn.config(state=tk.DISABLED)
            

            
        # 加载标签按钮只要有图片就可以启用
        if self.current_image_path:
            self.load_labels_btn.config(state=tk.NORMAL)
        else:
            self.load_labels_btn.config(state=tk.DISABLED)
            


    def load_labels(self):
        """批量加载目录下的所有标签，自动识别是分割标签还是检测标签"""
        # 弹出对话框让用户选择标签目录
        label_dir = filedialog.askdirectory(title="选择标签所在目录")
        
        if not label_dir:
            return
        
        try:
            # 获取目录下所有的.txt文件
            label_files = [f for f in os.listdir(label_dir) if f.lower().endswith('.txt')]
            if not label_files:
                messagebox.showinfo("提示", "所选目录中没有找到.txt格式的标签文件")
                return
            
            # 检查是否已经加载了图片
            if not self.current_image_path:
                messagebox.showwarning("警告", "请先加载图片")
                return
            
            # 获取所有已加载的图像路径
            if hasattr(self, 'image_list') and self.image_list:  # 批量加载模式
                loaded_images = self.image_list.copy()
            else:  # 单张图片模式
                loaded_images = [self.current_image_path]
            
            total_loaded = 0
            processed_images = 0
            
            # 首先尝试按文件名匹配
            for label_file in label_files:
                # 获取标签文件名（不包含扩展名）
                label_name = os.path.splitext(label_file)[0]
                
                # 查找匹配的图像文件
                matched_image = None
                for img_path in loaded_images:
                    img_name = os.path.splitext(os.path.basename(img_path))[0]
                    if img_name == label_name:
                        matched_image = img_path
                        break
                
                if matched_image:
                    # 处理匹配的图像和标签
                    loaded_count = self._process_label_file(
                        os.path.join(label_dir, label_file), 
                        matched_image
                    )
                    total_loaded += loaded_count
                    processed_images += 1
                    # 从列表中移除已处理的图像
                    loaded_images.remove(matched_image)
                    
                    # 如果所有图像都已处理，退出循环
                    if not loaded_images:
                        break
            
            # 确保image_annotations字典已初始化
            if not hasattr(self, 'image_annotations'):
                self.image_annotations = {}
            
            # 处理剩余的标签文件和图像（按顺序匹配）
            # 获取已处理过的图像名称列表
            # 已处理的图像是指从原始loaded_images中移除的那些
            original_loaded_images = [self.current_image_path] if not hasattr(self, 'image_list') or not self.image_list else self.image_list
            processed_image_names = [os.path.splitext(os.path.basename(img))[0] for img in original_loaded_images if img not in loaded_images]
            remaining_label_files = [f for f in label_files if os.path.splitext(f)[0] not in processed_image_names]
            
            min_count = min(len(remaining_label_files), len(loaded_images))
            for i in range(min_count):
                loaded_count = self._process_label_file(
                    os.path.join(label_dir, remaining_label_files[i]), 
                    loaded_images[i]
                )
                total_loaded += loaded_count
                processed_images += 1
            
            # 更新状态和显示
            self.status_var.set(f"成功批量加载{total_loaded - 1}个标注，共处理{processed_images}个图像")
            
            # 如果当前有图像显示，刷新显示
            if self.current_image_path:
                self._display_image()
                self._check_enable_save_buttons()
            
            messagebox.showinfo("成功", f"已成功批量加载标签\n共处理图像数: {processed_images}\n共加载标注数: {total_loaded}")
        except Exception as e:
            messagebox.showerror("错误", f"加载标签失败: {str(e)}")
            
    def _process_label_file(self, label_file_path, image_path):
        """处理单个标签文件并加载到对应的图像"""
        try:
            # 加载原始图像以获取尺寸
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # 读取标签文件
            with open(label_file_path, 'r') as f:
                lines = f.readlines()
            
            # 清除该图像的现有标注
            self.image_annotations[image_path] = []
            loaded_count = 0
            
            # 解析每一行标签
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 分割行数据
                parts = line.split()
                if len(parts) < 5:  # 至少需要类别索引和坐标信息
                    continue
                
                # 获取类别索引
                class_id = int(parts[0])
                
                # 获取类别名称
                class_name = f"类别{class_id}"
                if self.class_names and class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                
                # 创建掩码
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                
                # 自动识别标签类型
                # 分割标签: 通常有多个坐标点（点数>4）
                # 检测标签: 通常有4个值（中心x, 中心y, 宽度, 高度）
                if len(parts) == 5:  # 可能是检测标签（YOLO格式）
                    try:
                        # 解析边界框信息
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # 转换为实际像素坐标
                        x_min = int((center_x - width/2) * img_width)
                        y_min = int((center_y - height/2) * img_height)
                        x_max = int((center_x + width/2) * img_width)
                        y_max = int((center_y + height/2) * img_height)
                        
                        # 绘制矩形边界框到掩码
                        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 1, -1)
                        
                        # 添加到标注中
                        self.image_annotations[image_path].append({
                            'mask': mask,
                            'class_id': class_id,
                            'class_name': class_name
                        })
                        loaded_count += 1
                    except:
                        # 如果解析失败，尝试作为分割标签处理
                        pass
                
                # 尝试作为分割标签处理（多边形顶点）
                if len(self.image_annotations[image_path]) == loaded_count:  # 前面没有成功添加
                    try:
                        # 解析多边形顶点坐标
                        points = []
                        for i in range(1, len(parts), 2):
                            if i+1 < len(parts):
                                # 坐标是归一化的，需要转换回原始尺寸
                                norm_x = float(parts[i])
                                norm_y = float(parts[i+1])
                                x = int(norm_x * img_width)
                                y = int(norm_y * img_height)
                                points.append((x, y))
                        
                        # 如果有足够的点，绘制多边形到掩码
                        if len(points) >= 3:
                            pts = np.array(points, dtype=np.int32)
                            cv2.fillPoly(mask, [pts], 1)
                            
                            # 添加到标注中
                            self.image_annotations[image_path].append({
                                'mask': mask,
                                'class_id': class_id,
                                'class_name': class_name
                            })
                            loaded_count += 1
                    except:
                        # 解析失败，跳过此行
                        continue
            
            # 如果当前正在显示这个图像，更新当前图像的标注
            if image_path == self.current_image_path:
                self.current_image_annotations = self.image_annotations[image_path].copy()
            
            return loaded_count
        except Exception as e:
            print(f"处理标签文件 {os.path.basename(label_file_path)} 时出错: {str(e)}")
            return 0
    
    def on_canvas_right_click(self, event):
        # 优先检查是否右键点击在已分割区域上
        if self.hovered_mask_index != -1:
            # 获取要移除的掩码信息
            removed_annotation = self.current_image_annotations.pop(self.hovered_mask_index)
            self.hovered_mask_index = -1  # 重置悬停状态
            self.status_var.set(f"已取消分割区域: {removed_annotation['class_name']}")
            
            # 关键修复：清除self.masks变量，确保没有任何掩码会被绘制
            self.masks = None
            self.scores = None
            
            # 立即刷新显示，确保掩码消失
            self._display_image()
            
            # 强制更新所有UI组件
            self.root.update_idletasks()
            self.root.update()  # 强制完全更新
            
            # 重新绑定画布事件以确保事件处理器响应最新状态
            self.canvas.unbind("<Motion>")
            self.canvas.bind("<Motion>", self.on_canvas_motion)
            
            # 确保更新分割按钮状态
            self._check_enable_segment()
        elif self.points:
            # 如果有选中的锚点，移除它
            if self.selected_point_index != -1:
                removed_point = self.points.pop(self.selected_point_index)
                self.labels.pop(self.selected_point_index)
                self.selected_point_index = -1
                self.status_var.set(f"已取消选中的锚点，当前锚点数量: {len(self.points)}")
            else:
                # 否则移除最后一个锚点
                removed_point = self.points.pop()
                self.labels.pop()
                self.status_var.set(f"已取消上一个锚点，当前锚点数量: {len(self.points)}")
            
            # 刷新显示
            self._update_points_display()
            self._display_image()  # 重新显示图片
            self._check_enable_segment()
        else:
            self.status_var.set("没有可取消的锚点或分割区域")
    
    def on_zoom(self, event):
        # Ctrl+滚轮缩放功能
        scale_factor = 1.1  # 缩放因子
        
        # 计算原始图像的尺寸
        img_height, img_width = self.image.shape[:2]
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # 计算新的缩放比例
        if event.delta > 0:
            # 滚轮向上，放大
            new_scale = self.img_scale * scale_factor
        else:
            # 滚轮向下，缩小
            new_scale = self.img_scale / scale_factor
        
        # 设置缩放范围限制：最大10倍，不设置最小缩放限制
        max_scale = 10.0  # 最大缩放10倍
        # 仅限制最大缩放比例，不限制缩小比例
        new_scale = min(new_scale, max_scale)  # 仅限制最大缩放比例，不限制缩小
        
        if self.img_scale != new_scale:
            # 计算缩放前后的鼠标位置（始终使用鼠标位置作为参考点）
            x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            
            # 计算鼠标在图像上的原始坐标（缩放前）
            img_x = (x - self.img_offset[0]) / self.img_scale
            img_y = (y - self.img_offset[1]) / self.img_scale
            
            # 调整缩放比例
            self.img_scale = new_scale
            
            # 计算新的偏移量，使鼠标位置保持在缩放后的相同图像点上
            # 这确保了缩放始终围绕鼠标位置进行
            new_offset_x = x - img_x * self.img_scale
            new_offset_y = y - img_y * self.img_scale
            
            # 对于缩小操作，添加平滑向中心过渡的效果
            if event.delta < 0:
                # 计算新尺寸
                new_width = int(img_width * self.img_scale)
                new_height = int(img_height * self.img_scale)
                
                # 计算目标居中偏移量
                target_offset = ((canvas_width - new_width) // 2, (canvas_height - new_height) // 2)
                
                # 根据缩放比例调整平滑程度：缩放比例越小，越倾向于居中
                # 当接近原始尺寸时，逐渐增加居中权重
                center_weight = 0
                if self.img_scale < 2.0:
                    # 缩放比例在1.0-2.0之间时，权重从0线性增加到0.6
                    center_weight = min(0.6, (2.0 - self.img_scale) / 1.0 * 0.6)
                
                # 在保持鼠标位置和居中位置之间进行平滑过渡
                new_offset_x = int(new_offset_x * (1 - center_weight) + target_offset[0] * center_weight)
                new_offset_y = int(new_offset_y * (1 - center_weight) + target_offset[1] * center_weight)
            
            self.img_offset = (new_offset_x, new_offset_y)
            
            # 重新显示图像
            self._display_image()
            self._update_points_display()
            
            # 更新状态栏显示当前缩放比例
            if abs(self.img_scale - 1.0) < 0.01:
                self.status_var.set(f"缩放比例: 原始尺寸 (1.00x)")
            else:
                self.status_var.set(f"缩放比例: {self.img_scale:.2f}x")
    
    def on_canvas_configure(self, event):
        # 画布大小变化时重新显示图像
        if self.image is not None:
            self._display_image()
    
    def on_canvas_motion(self, event):
        # 检查鼠标是否靠近任何锚点或在已分割区域上
        if self.image is None:
            return
        
        # 获取鼠标在画布上的位置
        mouse_x, mouse_y = event.x, event.y
        
        # 检查锚点悬停
        closest_point_index = -1
        min_distance = 30  # 像素距离阈值
        
        # 检查每个锚点
        for i, (x, y) in enumerate(self.points):
            # 将图像坐标转换为画布坐标
            canvas_x = int(x * self.img_scale) + self.img_offset[0]
            canvas_y = int(y * self.img_scale) + self.img_offset[1]
            
            # 计算鼠标与锚点的距离
            distance = ((mouse_x - canvas_x) **2 + (mouse_y - canvas_y)** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_point_index = i
        
        # 检查掩码悬停
        hovered_mask_index = -1
        img_height, img_width = self.image.shape[:2]
        
        # 计算实际图像坐标
        img_x = int((mouse_x - self.img_offset[0]) / self.img_scale)
        img_y = int((mouse_y - self.img_offset[1]) / self.img_scale)
        
        # 确保坐标在图像范围内
        img_x = min(max(img_x, 0), img_width - 1)
        img_y = min(max(img_y, 0), img_height - 1)
        
        # 检查是否在已分割区域上
        for i, annotation in enumerate(self.current_image_annotations):
            mask = annotation['mask']
            # 处理掩码维度，确保它是二维的
            if mask.ndim == 3:
                if mask.shape[0] == 1:  # 移除通道维度
                    mask = mask[0]
                else:
                    mask = mask.squeeze()
            
            # 调整掩码尺寸以匹配原图
            if mask.shape[:2] != (img_height, img_width):
                mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            
            # 使用更敏感的阈值检查该点是否在掩码内
            if mask[img_y, img_x] > 0.2:  # 降低阈值以提高检测灵敏度
                hovered_mask_index = i
                break
        
        # 如果选中的锚点或悬停的掩码发生变化，更新状态并重新显示
        if closest_point_index != self.selected_point_index or hovered_mask_index != self.hovered_mask_index:
            self.selected_point_index = closest_point_index
            self.hovered_mask_index = hovered_mask_index
            self._display_image()

    def _update_points_display(self):
        self.points_text.delete(1.0, tk.END)
        for i, (x, y) in enumerate(self.points):
            self.points_text.insert(tk.END, f"点 {i+1}: ({x}, {y})\n")
    
    def _display_image(self):
        if self.image is None:
            return
        
        # 确保获取最新的画布尺寸
        self.canvas.update_idletasks()
        
        # 创建图像的副本进行显示
        display_img = self.image.copy()
        # 由于OpenCV操作使用BGR格式，而我们的图像是RGB格式，需要转换
        display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
        
        # 调整图像大小以适应画布
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # 确保最小尺寸以避免除以零
        canvas_width = max(canvas_width, 1)
        canvas_height = max(canvas_height, 1)
        
        img_height, img_width = display_img.shape[:2]
        
        # 如果是首次显示图像（img_scale未设置），则自动计算缩放比例使其正好占满视窗
        if not hasattr(self, 'img_scale') or self.img_scale is None:
            # 计算适应画布的缩放比例，使其正好占满视窗并保持原图比例
            scale_width = canvas_width / img_width
            scale_height = canvas_height / img_height
            self.img_scale = max(scale_width, scale_height)  # 取较大的比例，确保正好占满视窗
            
        # 使用当前缩放比例计算新尺寸
        new_width = int(img_width * self.img_scale)
        new_height = int(img_height * self.img_scale)
        
        # 计算偏移量，仅在首次显示时使图像居中
        if not hasattr(self, 'has_displayed') or not self.has_displayed:
            self.img_offset = ((canvas_width - new_width) // 2, (canvas_height - new_height) // 2)
            self.has_displayed = True
        
        # 调整图像大小
        display_img = cv2.resize(display_img, (new_width, new_height))
        
        # 绘制点
        for i, (x, y) in enumerate(self.points):
            # 根据是否被选中确定颜色
            if i == self.selected_point_index:
                color = (0, 255, 0)  # 绿色点，表示可选
            else:
                color = (0, 0, 255)  # 红色点
            # 由于我们是在调整大小后的图像上绘制点，所以直接使用原始坐标乘以缩放比例
            scaled_x = int(x * self.img_scale)
            scaled_y = int(y * self.img_scale)
            cv2.circle(display_img, (scaled_x, scaled_y), 8, color, -1)
            cv2.putText(display_img, f"{i+1}", (scaled_x+10, scaled_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 为每个类别生成固定颜色
        class_colors = {}
        for annotation in self.current_image_annotations:
            class_name = annotation['class_name']
            if class_name not in class_colors:
                # 使用类名的哈希值生成一致的颜色
                color_index = hash(class_name) % 100
                hue = color_index * (137.5 / 360)
                rgb = plt.cm.hsv(hue)
                class_colors[class_name] = np.array([rgb[0], rgb[1], rgb[2]])
        
        # 绘制所有保存的掩码和对应的矩形框
        for i, annotation in enumerate(self.current_image_annotations):
            mask = annotation['mask']
            # 确保掩码是二维的
            if mask.ndim == 3:
                if mask.shape[0] == 1:
                    mask = mask[0]
                else:
                    mask = mask.squeeze()
            
            # 确保掩码数据类型正确
            mask = mask.astype(np.uint8)
            class_name = annotation['class_name']
            base_color = class_colors[class_name]  # 基础颜色
            
            # 定义选中和非选中状态的颜色
            if i == self.hovered_mask_index:
                # 悬停状态：使用暗绿色表示选中
                dark_green = np.array([0.0, 0.5, 0.0])  # 暗绿色RGB值
                color = dark_green
            else:
                # 非悬停状态：保持原有类别的基础颜色不变
                color = base_color
            
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255
            mask_image = mask_image.astype(np.uint8)
            
            # 调整掩码图像尺寸以匹配显示图像
            mask_image = cv2.resize(mask_image, (display_img.shape[1], display_img.shape[0]))
            
            # 当鼠标悬停在该掩码上时，加深颜色（使用更高的alpha值）
            alpha = 0.8 if i == self.hovered_mask_index else 0.5
            display_img = cv2.addWeighted(display_img, 1, mask_image, alpha, 0)
            
            # 绘制保存的矩形框（如果有）
            if 'bbox' in annotation and annotation['bbox'] is not None:
                x_min, y_min, x_max, y_max = annotation['bbox']
                # 调整坐标以匹配显示图像的尺寸
                scaled_x_min = int(x_min * self.img_scale)
                scaled_y_min = int(y_min * self.img_scale)
                scaled_x_max = int(x_max * self.img_scale)
                scaled_y_max = int(y_max * self.img_scale)
                # 使用已经为该类别生成的颜色
                bgr_color = (int(base_color[2] * 255), int(base_color[1] * 255), int(base_color[0] * 255))
                # 绘制矩形框（使用与掩码相同的颜色，线宽为2）
                cv2.rectangle(display_img, (scaled_x_min, scaled_y_min), (scaled_x_max, scaled_y_max), bgr_color, 2)
        
        # 绘制当前掩码（如果有）
        if self.masks is not None:
            mask = self.masks[0]  # 使用得分最高的掩码
            color = np.array([30/255, 144/255, 255/255])  # 使用蓝色显示当前掩码
            h, w = mask.shape[-2:]
            mask = mask.astype(np.uint8)
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255
            mask_image = mask_image.astype(np.uint8)
            
            # 调整掩码图像尺寸以匹配显示图像
            mask_image = cv2.resize(mask_image, (display_img.shape[1], display_img.shape[0]))
            
            # 将掩码叠加到图像上
            display_img = cv2.addWeighted(display_img, 1, mask_image, 0.5, 0)
        
        # 绘制当前掩码的最小矩形框（如果有）
        if self.current_mask_bbox is not None and self.masks is not None:
            x_min, y_min, x_max, y_max = self.current_mask_bbox
            # 调整坐标以匹配显示图像的尺寸
            scaled_x_min = int(x_min * self.img_scale)
            scaled_y_min = int(y_min * self.img_scale)
            scaled_x_max = int(x_max * self.img_scale)
            scaled_y_max = int(y_max * self.img_scale)
            # 绘制矩形框（使用蓝色，线宽为2）
            cv2.rectangle(display_img, (scaled_x_min, scaled_y_min), (scaled_x_max, scaled_y_max), (255, 0, 0), 2)
        
        # 将BGR格式转换回RGB格式以在Tkinter中正确显示
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        self.display_img = ImageTk.PhotoImage(image=Image.fromarray(display_img))
        
        # 在画布上显示图像
        self.canvas.delete("all")
        self.canvas.create_image(self.img_offset[0], self.img_offset[1], image=self.display_img, anchor=tk.NW)
    
    def _check_enable_segment(self):
        if self.image is not None and self.predictor is not None and len(self.points) > 0:
            self.segment_btn.config(state=tk.NORMAL)
        else:
            self.segment_btn.config(state=tk.DISABLED)
    
    def perform_segmentation(self):
        if self.image is None or self.predictor is None:
            messagebox.showerror("错误", "请先加载图片和模型")
            return
        
        if len(self.points) < 1:
            messagebox.showerror("错误", "请至少选择一个点")
            return
        
        try:
            self.status_var.set("正在执行分割...")
            self.root.update()
            
            # 设置图像
            self.predictor.set_image(self.image)
            
            # 准备点数据
            input_point = np.array(self.points)
            input_label = np.array(self.labels)
            
            # 检查是否有已分割的区域需要排除
            if self.current_image_annotations:
                # 为每个已分割区域的中心添加一个负点标签（值为0），表示排除的区域
                for annotation in self.current_image_annotations:
                    mask = annotation['mask']
                    # 找到掩码的中心点
                    coords = np.column_stack(np.where(mask))
                    if coords.size > 0:
                        center = coords.mean(axis=0).astype(int)
                        # 将中心点添加为负点
                        input_point = np.vstack([input_point, [center[1], center[0]]])  # 注意这里是 [x, y]
                        input_label = np.append(input_label, 0)  # 0表示负点
            
            # 注意：取消锚点后，用户可以重新选择锚点进行分割，
            # 这种情况下已保存的区域仍然会被锁定，但用户可以通过删除整个标注来解除锁定
            
            # 执行预测 - 设置normalize_coords=True让predictor处理坐标归一化
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
                normalize_coords=True
            )
            
            # 按得分排序
            sorted_ind = np.argsort(scores)[::-1]
            self.masks = masks[sorted_ind]
            self.scores = scores[sorted_ind]
            
            # 确保分割处理当前缩放尺寸下的工作区域
            # 获取当前画布显示区域的像素边界
            h, w = self.image.shape[:2]
            
            # 计算当前视图在原始图像中的边界
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # 计算视图区域在原始图像中的坐标范围
            view_x_min = max(0, int((-self.img_offset[0]) / self.img_scale))
            view_y_min = max(0, int((-self.img_offset[1]) / self.img_scale))
            view_x_max = min(w, int((-self.img_offset[0] + canvas_width) / self.img_scale))
            view_y_max = min(h, int((-self.img_offset[1] + canvas_height) / self.img_scale))
            
            # 在状态栏显示当前分割范围信息
            self.status_var.set(f"分割处理区域: x({view_x_min}-{view_x_max}), y({view_y_min}-{view_y_max})，缩放比例: {self.img_scale:.2f}x")
            
            # 实际分割时，我们通过修改掩码后处理来限制结果范围
            # 确保掩码只在当前视图范围内有效
            view_masks = []
            for mask in self.masks:
                # 创建一个与原掩码相同大小的零矩阵
                view_mask = np.zeros_like(mask)
                # 只保留视图范围内的掩码值
                if mask.ndim == 3:
                    view_mask[:, view_y_min:view_y_max, view_x_min:view_x_max] = mask[:, view_y_min:view_y_max, view_x_min:view_x_max]
                else:
                    view_mask[view_y_min:view_y_max, view_x_min:view_x_max] = mask[view_y_min:view_y_max, view_x_min:view_x_max]
                view_masks.append(view_mask)
            
            # 使用限制后的掩码
            self.masks = np.array(view_masks)
            
            # 保存结果到缓存
            if self.current_image_path:
                self.image_results[self.current_image_path] = {
                    'masks': self.masks,
                    'scores': self.scores,
                    'points': self.points.copy(),
                    'labels': self.labels.copy()
                }
            
            self.status_var.set(f"分割完成，最高得分: {self.scores[0]:.3f}")
            self.save_all_btn.config(state=tk.NORMAL)
            self._check_enable_save_object()
            
            # 计算并保存覆盖整个掩码的最小矩形框
            if self.masks is not None and len(self.masks) > 0:
                # 获取得分最高的掩码
                mask = self.masks[0]
                # 找到掩码中的所有非零像素坐标
                coords = np.column_stack(np.where(mask > 0.5))
                if coords.size > 0:
                    # 计算最小外接矩形
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    # 保存矩形框坐标
                    self.current_mask_bbox = (x_min, y_min, x_max, y_max)
                else:
                    self.current_mask_bbox = None
            else:
                self.current_mask_bbox = None
            
            # 显示结果
            self._display_image()
        except Exception as e:
            messagebox.showerror("错误", f"分割失败: {str(e)}")
            self.status_var.set("分割失败")
    
    def save_current_object(self):
        """保存当前分割的目标到标注列表"""
        if self.masks is None:
            messagebox.showerror("错误", "没有可保存的分割结果")
            return
        
        # 创建类别选择对话框
        class_name = self._show_class_selection_dialog()
        if not class_name:
            return
        
        # 为新类别分配ID
        if class_name not in self.class_to_id:
            self.class_names.append(class_name)
            self.class_to_id[class_name] = len(self.class_to_id)
        
        class_id = self.class_to_id[class_name]
        
        # 保存当前掩码、类别信息和矩形框
        annotation = {
            'mask': self.masks[0].copy(),
            'class_id': class_id,
            'class_name': class_name,
            'mask_id': self.current_mask_id,
            'bbox': self.current_mask_bbox  # 保存矩形框信息
        }
        
        self.current_image_annotations.append(annotation)
        
        # 同时更新到 image_annotations 字典中，确保可以通过 save_all_results 保存
        # 使用 current_image_path 而不是 image_path，确保路径一致
        if self.current_image_path not in self.image_annotations:
            self.image_annotations[self.current_image_path] = []
        self.image_annotations[self.current_image_path].append(annotation)
        
        # 更新保存按钮状态
        self._check_enable_save_buttons()
        
        self.current_mask_id += 1
        
        # 显示已保存的所有掩码
        self._display_image()
        
        # 保存目标后锚点消失，让用户开始新的分割
        self.points = []
        self.labels = []
        self.selected_point_index = -1
        self.hovered_mask_index = -1  # 重置悬停的掩码索引
        self.current_mask_bbox = None  # 清除当前的矩形框
        self._update_points_display()
        self._check_enable_segment()
        
        self.status_var.set(f"已保存目标: {class_name}，锚点已重置，可开始新的分割")
        self.save_object_btn.config(state=tk.DISABLED)
        self.save_all_btn.config(state=tk.NORMAL)
        
    def _show_class_selection_dialog(self):
        """显示类别选择对话框"""
        # 获取当前鼠标位置
        x = self.root.winfo_pointerx()
        y = self.root.winfo_pointery()
        
        # 创建对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("选择/创建类别")
        dialog.geometry("400x350+%d+%d" % (x, y))
        dialog.transient(self.root)
        
        # 加强模态性，确保对话框捕获所有事件，防止主程序快捷键被触发
        dialog.grab_set()  # 捕获所有事件到对话框
        dialog.focus_force()  # 强制获取焦点
        
        # 创建结果存储变量
        result = [None]  # 用于存储选择的类别名称
        
        # 创建输入区域
        input_frame = ttk.Frame(dialog)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(input_frame, text="输入对象标签").pack(anchor="w")
        
        # 输入框
        class_entry_var = tk.StringVar()
        class_entry = ttk.Entry(input_frame, textvariable=class_entry_var, width=30)
        class_entry.pack(fill=tk.X, pady=5)
        class_entry.focus_set()
        
        # 重要：返回"break"阻止键盘事件传播到主窗口，但不会影响输入框的文本输入
        def handle_key_press(event):
            # 返回"break"可以完全阻止键盘事件传播到主窗口
            # 这样在输入对象标签时包含快捷键字母就不会触发主程序的按钮了
            return "break"
        
        # 只为对话框绑定键盘事件处理，让输入框可以正常接收输入
        dialog.bind("<KeyPress>", handle_key_press)
        
        # 创建类别列表区域
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 如果有类别，创建类别列表区域
        if self.class_names:
            tk.Label(list_frame, text="或选择已有类别：").pack(anchor="w", pady=5)
            
            # 创建滚动区域
            canvas = tk.Canvas(list_frame, height=150)
            scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # 创建类别按钮
            for name in self.class_names:
                btn_frame = ttk.Frame(scrollable_frame)
                btn_frame.pack(fill=tk.X, pady=2, padx=5)
                
                # 选择按钮
                def select_class(n=name):
                    class_entry_var.set(n)
                    
                select_btn = ttk.Button(
                    btn_frame, 
                    text=name, 
                    command=select_class, 
                    width=20
                )
                select_btn.pack(side=tk.LEFT, padx=5)
                
                # 重命名按钮
                def rename_class(n=name):
                    # 创建重命名对话框
                    rename_dialog = tk.Toplevel(dialog)
                    rename_dialog.title("重命名类别")
                    rename_dialog.geometry("300x150")
                    rename_dialog.transient(dialog)
                    rename_dialog.grab_set()
                    rename_dialog.focus_force()
                    
                    # 添加键盘事件处理，阻止事件传播到主窗口
                    def handle_rename_key_press(event):
                        # 返回"break"可以完全阻止键盘事件传播到主窗口
                        # 这样在输入新名称时包含快捷键字母就不会触发主程序的按钮了
                        return "break"
                    
                    # 只为重命名对话框绑定键盘事件处理
                    rename_dialog.bind("<KeyPress>", handle_rename_key_press)
                    
                    tk.Label(rename_dialog, text="新名称：").pack(pady=10)
                    new_name_var = tk.StringVar(value=n)
                    new_name_entry = ttk.Entry(rename_dialog, textvariable=new_name_var, width=20)
                    new_name_entry.pack(pady=5)
                    new_name_entry.focus_set()
                    new_name_entry.select_range(0, tk.END)
                    
                    def do_rename():
                        new_name = new_name_var.get().strip()
                        if new_name and new_name != n:
                            # 更新类别信息
                            class_id = self.class_to_id.pop(n)
                            self.class_names.remove(n)
                            self.class_to_id[new_name] = class_id
                            self.class_names.append(new_name)
                            
                            # 更新现有标注中的类别名称
                            for annotation in self.current_image_annotations:
                                if annotation['class_name'] == n:
                                    annotation['class_name'] = new_name
                            
                            # 更新输入框和刷新界面
                            class_entry_var.set(new_name)
                            
                            # 刷新类别列表显示
                            # 首先清除现有列表
                            for widget in scrollable_frame.winfo_children():
                                widget.destroy()
                            # 然后重新创建类别按钮
                            for name in self.class_names:
                                btn_frame = ttk.Frame(scrollable_frame)
                                btn_frame.pack(fill=tk.X, pady=2, padx=5)
                                
                                # 选择按钮
                                def select_class(n=name):
                                    class_entry_var.set(n)
                                
                                select_btn = ttk.Button(
                                    btn_frame, 
                                    text=name, 
                                    command=select_class, 
                                    width=20
                                )
                                select_btn.pack(side=tk.LEFT, padx=5)
                                
                                # 重命名按钮
                                def rename_class(n=name):
                                    # 创建重命名对话框
                                    rename_dialog = tk.Toplevel(dialog)
                                    rename_dialog.title("重命名类别")
                                    rename_dialog.geometry("300x150")
                                    rename_dialog.transient(dialog)
                                    rename_dialog.grab_set()
                                    rename_dialog.focus_force()
                                    
                                    # 添加键盘事件处理，阻止事件传播到主窗口
                                    def handle_rename_key_press(event):
                                        # 返回"break"可以完全阻止键盘事件传播到主窗口
                                        # 这样在输入新名称时包含快捷键字母就不会触发主程序的按钮了
                                        return "break"
                                    
                                    # 只为重命名对话框绑定键盘事件处理
                                    rename_dialog.bind("<KeyPress>", handle_rename_key_press)
                                    
                                    tk.Label(rename_dialog, text="新名称：").pack(pady=10)
                                    new_name_var = tk.StringVar(value=n)
                                    new_name_entry = ttk.Entry(rename_dialog, textvariable=new_name_var, width=20)
                                    new_name_entry.pack(pady=5)
                                    new_name_entry.focus_set()
                                    new_name_entry.select_range(0, tk.END)
                                    
                                    def do_rename():
                                        new_name = new_name_var.get().strip()
                                        if new_name and new_name != n:
                                            # 更新类别信息
                                            class_id = self.class_to_id.pop(n)
                                            self.class_names.remove(n)
                                            self.class_to_id[new_name] = class_id
                                            self.class_names.append(new_name)
                                            
                                            # 更新现有标注中的类别名称
                                            for annotation in self.current_image_annotations:
                                                if annotation['class_name'] == n:
                                                    annotation['class_name'] = new_name
                                            
                                            # 更新输入框和刷新界面
                                            class_entry_var.set(new_name)
                                            rename_dialog.destroy()
                                        else:
                                            messagebox.showwarning("警告", "请输入有效的新名称")
                                    
                                    btn_frame = ttk.Frame(rename_dialog)
                                    btn_frame.pack(pady=10)
                                    
                                    ttk.Button(btn_frame, text="确定", command=do_rename).pack(side=tk.LEFT, padx=5)
                                    ttk.Button(btn_frame, text="取消", command=rename_dialog.destroy).pack(side=tk.LEFT, padx=5)
                                    
                                    rename_dialog.bind("<Return>", lambda event: do_rename())
                                    dialog.wait_window(rename_dialog)
                                
                                rename_btn = ttk.Button(
                                    btn_frame, 
                                    text="重命名", 
                                    command=rename_class, 
                                    width=10
                                )
                                rename_btn.pack(side=tk.LEFT, padx=5)
                            
                            # 刷新画布滚动区域
                            canvas.configure(scrollregion=canvas.bbox("all"))
                            
                            rename_dialog.destroy()
                        else:
                            messagebox.showwarning("警告", "请输入有效的新名称")
                    
                    btn_frame = ttk.Frame(rename_dialog)
                    btn_frame.pack(pady=10)
                    
                    ttk.Button(btn_frame, text="确定", command=do_rename).pack(side=tk.LEFT, padx=5)
                    ttk.Button(btn_frame, text="取消", command=rename_dialog.destroy).pack(side=tk.LEFT, padx=5)
                    
                    rename_dialog.bind("<Return>", lambda event: do_rename())
                    dialog.wait_window(rename_dialog)
                
                rename_btn = ttk.Button(
                    btn_frame, 
                    text="重命名", 
                    command=rename_class, 
                    width=10
                )
                rename_btn.pack(side=tk.LEFT, padx=5)
        
        # 按钮框架 - 始终位于对话框底部，使用固定的pack位置确保不被遮挡
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(side=tk.BOTTOM, pady=10, fill=tk.X, padx=10)
        
        def on_ok():
            selected = class_entry_var.get().strip()
            if selected:
                result[0] = selected
                dialog.destroy()
            else:
                messagebox.showwarning("警告", "请输入类别名称")
        
        def on_cancel():
            dialog.destroy()
        
        # 确定和取消按钮
        button_container = ttk.Frame(btn_frame)
        button_container.pack(fill=tk.X, pady=5)
        ttk.Button(button_container, text="确定", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_container, text="取消", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        # 按Enter键确认，按Escape键取消
        dialog.bind("<Return>", lambda event: on_ok())
        dialog.bind("<Escape>", lambda event: on_cancel())
        
        # 等待对话框关闭
        self.root.wait_window(dialog)
        return result[0]
        
    # 移除save_current_results方法
            
    def save_all_results(self):
        # 弹出对话框让用户选择保存目录
        save_dir = filedialog.askdirectory(title="选择保存目录")
        if not save_dir:
            return
        
        # 检查是否有任何图像有标注结果
        has_annotations = False
        for path, annotations in self.image_annotations.items():
            if annotations:
                has_annotations = True
                break
        
        if not has_annotations:
            messagebox.showerror("错误", "没有可保存的标注结果")
            return
        
        # 获取用户选择的数据集类型
        dataset_type = self.dataset_type.get()
        
        try:
            # 为每个有标注的图像保存结果
            saved_count = 0
            for image_path, annotations in self.image_annotations.items():
                if not annotations:
                    continue
                
                try:
                    # 获取图片名称（不包含扩展名）
                    img_name = os.path.splitext(os.path.basename(image_path))[0]
                    
                    # 绘制并保存分割图像
                    segmented_file_path = os.path.join(save_dir, f"{img_name}_segmented.png")
                    
                    # 加载原始图像
                    image = np.array(Image.open(image_path).convert("RGB"))
                    result_img = image.copy()
                    
                    # 保存为YOLO格式的标签文件
                    yolo_file_path = os.path.join(save_dir, f"{img_name}.txt")
                    
                    # 读取原始图像尺寸
                    original_img = Image.open(image_path)
                    img_width, img_height = original_img.size
                    
                    with open(yolo_file_path, "w") as f:
                        # 对每个标注生成YOLO格式数据
                        for annotation in annotations:
                            class_id = annotation['class_id']
                            
                            if dataset_type == "分割":
                                # 分割数据集 - 保存多边形顶点坐标
                                mask = annotation['mask'].astype(np.uint8)
                                
                                # 查找轮廓
                                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                # 只处理最大的轮廓或第一个轮廓，避免一个对象生成多个标签
                                if contours:
                                    # 选择最大的轮廓
                                    contour = max(contours, key=cv2.contourArea)
                                    
                                    if len(contour) >= 4:  # 需要至少4个点
                                        # 简化轮廓以减少点数
                                        epsilon = 0.001 * cv2.arcLength(contour, True)
                                        approx = cv2.approxPolyDP(contour, epsilon, True)
                                        
                                        # 构建多边形顶点的归一化坐标列表
                                        polygon_points = []
                                        for point in approx:
                                            x, y = point[0]
                                            # 归一化坐标到0-1范围
                                            norm_x = x / img_width
                                            norm_y = y / img_height
                                            polygon_points.append(f"{norm_x:.6f}")
                                            polygon_points.append(f"{norm_y:.6f}")
                                        
                                        # 写入分割标签格式的数据：类别 顶点坐标列表
                                        f.write(f"{class_id} {' '.join(polygon_points)}\n")
                            else:
                                # 检测数据集 - 只保存边界框
                                mask = annotation['mask'].astype(np.uint8)
                                
                                # 查找掩码中的所有非零像素坐标
                                coords = np.column_stack(np.where(mask > 0.5))
                                if coords.size > 0:
                                    # 计算边界框
                                    y_min, x_min = coords.min(axis=0)
                                    y_max, x_max = coords.max(axis=0)
                                    
                                    # 计算中心坐标和宽高，并归一化到0-1范围
                                    center_x = (x_min + x_max) / (2 * img_width)
                                    center_y = (y_min + y_max) / (2 * img_height)
                                    width = (x_max - x_min) / img_width
                                    height = (y_max - y_min) / img_height
                                    
                                    # 写入YOLO格式的数据：类别 中心x 中心y 宽度 高度
                                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                
                    # 绘制所有掩码到分割图像
                    colors = self._generate_colors(len(annotations))
                    for i, annotation in enumerate(annotations):
                        mask = annotation['mask'].astype(np.uint8)
                        color = colors[i]  # 使用预先生成的颜色
                        
                        h, w = mask.shape[-2:]
                        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255
                        mask_image = mask_image.astype(np.uint8)
                        
                        # 确保掩码图像尺寸与原始图像匹配
                        mask_image = cv2.resize(mask_image, (result_img.shape[1], result_img.shape[0]))
                        
                        # 将掩码叠加到图像上
                        result_img = cv2.addWeighted(result_img, 1, mask_image, 0.5, 0)
                        
                    # 保存分割图像
                    Image.fromarray(result_img).save(segmented_file_path)
                    saved_count += 1
                except Exception as img_err:
                    messagebox.showwarning("警告", f"处理图像 {os.path.basename(image_path)} 时出错: {str(img_err)}")
                    continue
            
            # 保存类别映射文件
            classes_file_path = os.path.join(save_dir, "classes.txt")
            if self.class_names:
                with open(classes_file_path, "w") as f:
                    for class_name in self.class_names:
                        f.write(f"{self.class_to_id[class_name]} {class_name}\n")
            
            messagebox.showinfo("成功", f"已完成所有保存，共保存了 {saved_count} 个图像的标注结果\n数据集类型: {dataset_type}\n类别映射保存在:\n{classes_file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
        
    def _generate_colors(self, num_colors):
        """生成指定数量的不同颜色"""
        colors = []
        for i in range(num_colors):
            # 使用HSV颜色空间，确保颜色分布均匀
            hue = i * (137.5 / 360)  # 使用黄金角度分割，获得更多样化的颜色
            saturation = 0.8
            value = 0.9
            
            # 转换为RGB
            rgb = plt.cm.hsv(hue)
            colors.append(np.array([rgb[0], rgb[1], rgb[2]]))
        return colors
        
    def reset_app(self):
        # 清除锚点、分割结果和缓存
        self.points = []
        self.labels = []
        self.masks = None
        self.scores = None
        self.current_point_index = 0
        self.current_image_annotations = []
        self.current_mask_id = 0
        
        # 清除当前图片的缓存结果
        if self.current_image_path and self.current_image_path in self.image_results:
            del self.image_results[self.current_image_path]
        
        self.canvas.delete("all")
        self.points_text.delete(1.0, tk.END)
        self.status_var.set("已重置锚点和分割结果")
        self._display_image()  # 重新显示图片

        self.save_object_btn.config(state=tk.DISABLED)
        self.save_all_btn.config(state=tk.DISABLED)
        self.save_object_btn.config(state=tk.DISABLED)

        self.points = []
        self.labels = []
        self.current_point_index = 0
        self.segmented_image = None
        self.masks = None
        self.scores = None
        self.points_text.delete(1.0, tk.END)
        
        # 清除当前图片的缓存结果
        if self.current_image_path and self.current_image_path in self.image_results:
            del self.image_results[self.current_image_path]
        
        self.status_var.set("已重置锚点和分割结果")
        self.canvas.delete("all")
        self._display_image()
        self.segment_btn.config(state=tk.DISABLED)
        self.save_all_btn.config(state=tk.DISABLED)

# 运行应用
if __name__ == "__main__":
    root = tk.Tk()
    app = SAMInteractiveApp(root)
    # 监听窗口大小变化，重新显示图像
    def on_resize(event):
        # 防止窗口初始化时的事件触发
        if event.widget == root and event.widget.winfo_width() > 1 and event.widget.winfo_height() > 1:
            app._display_image()
    root.bind("<Configure>", on_resize)
    root.mainloop()
