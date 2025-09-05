
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

class EnhancedImageCropper:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("阿君蜜汁小工具-Crop")
        self.crop_coords = None
        self.original_image = None
        self.displayed_image = None
        self.image_files = []
        self.current_index = 0
        self.output_dir = "cropped_results"
        self.coords_dir = "saved_coords"
        self.rect = None
        self.locked = False
        self.drawing = False
        self.scale_factor = 1.0
        self.start_x = None
        self.start_y = None
        
        self.setup_ui()
        self.bind_events()
        self.root.mainloop()
    
    def setup_ui(self):
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg='white')
        self.canvas.pack()
        
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="选择文件夹(X)", command=self.select_folder).pack(side=tk.LEFT, padx=5)
        self.draw_btn = tk.Button(btn_frame, text="绘制裁剪框(C)", command=self.start_drawing)
        self.draw_btn.pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="锁定/解锁(S)", command=self.toggle_lock).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="上一张(A)", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="下一张(D)", command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        coord_frame = tk.Frame(self.root)
        coord_frame.pack(pady=5)
        tk.Button(coord_frame, text="保存坐标到文件", command=self.save_coords_to_file).pack(side=tk.LEFT, padx=5)
        self.load_btn = tk.Button(coord_frame, text="从文件加载坐标(L)", command=self.load_coords_from_file)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.dpi_label = tk.Label(self.root, text="DPI: -- | 缩放比例: 1.0x")
        self.dpi_label.pack()
        
        self.status_label = tk.Label(self.root, text="状态: 未锁定 | 绘制模式: 关闭 | 图片: 0/0", fg="black")
        self.status_label.pack()
    
    def bind_events(self):
        self.root.bind('x', lambda e: self.select_folder())
        self.root.bind('c', lambda e: self.start_drawing())
        self.root.bind('s', lambda e: self.toggle_lock())
        self.root.bind('a', lambda e: self.prev_image())
        self.root.bind('d', lambda e: self.next_image())
        self.root.bind('l', lambda e: self.load_coords_from_file())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.cancel_crop)
    
    def update_ui_state(self):
        if self.drawing:
            self.draw_btn.config(bg='green', fg='white')
        else:
            self.draw_btn.config(bg='SystemButtonFace', fg='black')
        
        if self.locked:
            self.status_label.config(fg='green')
        else:
            self.status_label.config(fg='red')
    
    def save_coords_to_file(self):
        if not self.crop_coords or not self.image_files:
            messagebox.showwarning("警告", "没有可保存的坐标数据")
            return
            
        if not os.path.exists(self.coords_dir):
            os.makedirs(self.coords_dir)
        
        img_name = os.path.basename(self.image_files[self.current_index])
        txt_name = os.path.splitext(img_name)[0] + "_coords.txt"
        txt_path = os.path.join(self.coords_dir, txt_name)
        
        with open(txt_path, 'w') as f:
            f.write(f"{self.crop_coords[0]},{self.crop_coords[1]},{self.crop_coords[2]},{self.crop_coords[3]}")
        
        messagebox.showinfo("成功", f"坐标已保存到:\n{txt_path}")
    
    def load_coords_from_file(self):
        if not self.image_files:
            messagebox.showwarning("警告", "请先选择图片")
            return
            
        img_name = os.path.basename(self.image_files[self.current_index])
        txt_name = os.path.splitext(img_name)[0] + "_coords.txt"
        txt_path = os.path.join(self.coords_dir, txt_name)
        
        if not os.path.exists(txt_path):
            messagebox.showwarning("警告", f"未找到对应的坐标文件:\n{txt_path}")
            return
            
        try:
            with open(txt_path, 'r') as f:
                coords = f.read().strip().split(',')
                if len(coords) == 4:
                    self.crop_coords = tuple(map(float, coords))
                    self.redraw_crop_box()
                    messagebox.showinfo("成功", "坐标已加载")
                else:
                    messagebox.showerror("错误", "坐标文件格式不正确")
        except Exception as e:
            messagebox.showerror("错误", f"读取坐标文件失败:\n{str(e)}")
    
    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_files = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))
            ]
            if self.image_files:
                self.current_index = 0
                self.load_image()
    
    def load_image(self):
        if 0 <= self.current_index < len(self.image_files):
            img_path = self.image_files[self.current_index]
            self.original_image = Image.open(img_path)
            
            dpi = self.original_image.info.get('dpi', (72, 72))
            self.dpi_label.config(text=f"DPI: {dpi[0]}x{dpi[1]} | 缩放比例: {self.scale_factor:.2f}x")
            
            self.display_image()
            self.redraw_crop_box()
    
    def display_image(self):
        self.canvas.delete("all")
        if not self.original_image:
            return
            
        img_width, img_height = self.original_image.size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        width_ratio = canvas_width / img_width
        height_ratio = canvas_height / img_height
        self.scale_factor = min(width_ratio, height_ratio)
        
        display_size = (int(img_width * self.scale_factor), int(img_height * self.scale_factor))
        self.displayed_image = self.original_image.copy()
        self.displayed_image.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(self.displayed_image)
        self.canvas.create_image(
            canvas_width//2, 
            canvas_height//2, 
            image=self.tk_image
        )
        self.update_status()
    
    def redraw_crop_box(self):
        if self.crop_coords:
            self.rect = self.canvas.create_rectangle(
                *self.crop_coords,
                outline='red', width=2, tags="crop_box"
            )
    
    def start_drawing(self):
        if not self.locked:
            self.drawing = True
            self.start_x = None
            self.start_y = None
            self.canvas.delete("crop_box")
            self.update_status()
            self.update_ui_state()
    
    def on_press(self, event):
        if self.drawing and not self.locked:
            self.start_x = event.x
            self.start_y = event.y
            self.rect = self.canvas.create_rectangle(
                self.start_x, self.start_y, 
                self.start_x, self.start_y,
                outline='red', width=2, tags="crop_box"
            )
    
    def on_drag(self, event):
        if self.rect and not self.locked:
            self.canvas.coords(
                self.rect, 
                self.start_x, self.start_y, 
                event.x, event.y
            )
    
    def on_release(self, event):
        if self.rect and not self.locked:
            self.crop_coords = (
                min(self.start_x, event.x),
                min(self.start_y, event.y),
                max(self.start_x, event.x),
                max(self.start_y, event.y)
            )
            self.drawing = False
            self.update_status()
            self.update_ui_state()
    
    def toggle_lock(self):
        self.locked = not self.locked
        self.update_status()
        self.update_ui_state()
    
    def cancel_crop(self, event):
        if not self.locked:
            self.canvas.delete("crop_box")
            self.crop_coords = None
            self.rect = None
            self.update_status()
    
    def update_status(self):
        status = "已锁定" if self.locked else "未锁定"
        draw_mode = "开启" if self.drawing else "关闭"
        color = "green" if self.locked else "red"
        count = f"{self.current_index + 1}/{len(self.image_files)}" if self.image_files else "0/0"
        self.status_label.config(
            text=f"状态: {status} | 绘制模式: {draw_mode} | 图片: {count}",
            fg=color
        )
    
    def auto_crop(self):
        if self.original_image and self.crop_coords:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_width, img_height = self.original_image.size
            
            display_width = int(img_width * self.scale_factor)
            display_height = int(img_height * self.scale_factor)
            offset_x = (canvas_width - display_width) // 2
            offset_y = (canvas_height - display_height) // 2
            
            x1 = int((self.crop_coords[0] - offset_x) / self.scale_factor)
            y1 = int((self.crop_coords[1] - offset_y) / self.scale_factor)
            x2 = int((self.crop_coords[2] - offset_x) / self.scale_factor)
            y2 = int((self.crop_coords[3] - offset_y) / self.scale_factor)
            
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))
            
            if x1 >= x2 or y1 >= y2:
                print("无效的裁剪区域")
                return
            
            cropped_img = self.original_image.crop((x1, y1, x2, y2))
            
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            original_name = os.path.basename(self.image_files[self.current_index])
            name_without_ext = os.path.splitext(original_name)[0]
            ext = os.path.splitext(original_name)[1].lower()
            
            output_path = os.path.join(
                self.output_dir, 
                f"{name_without_ext}_crop{ext}"
            )
            cropped_img.save(output_path)
            print(f"已保存裁剪图片到: {output_path}")
            print(f"实际裁剪尺寸: {cropped_img.size[0]}x{cropped_img.size[1]} 像素")
    
    def prev_image(self):
        if self.image_files:
            if self.crop_coords:
                self.auto_crop()
            
            if self.current_index > 0:
                self.current_index -= 1
                self.load_image()
    
    def next_image(self):
        if self.image_files:
            if self.crop_coords:
                self.auto_crop()
            
            if self.current_index < len(self.image_files) - 1:
                self.current_index += 1
                self.load_image()
            else:
                print("已处理完所有图片")

if __name__ == "__main__":
    EnhancedImageCropper()