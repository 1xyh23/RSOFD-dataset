import cv2
import json
import random
import numpy as np
import os
import torch
import sys
from pathlib import Path

from simple_lama_inpainting import SimpleLama
from PIL import Image


class ImageProcessor:
    def __init__(self, save_dir='output'):
        # 初始化保存路径
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    @staticmethod
    def load_image(image_path):
        """加载图像文件"""
        return cv2.imread(image_path)

    @staticmethod
    def load_annotations(annotation_path):
        """加载标注文件"""
        with open(annotation_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_image(image, output_path):
        """保存图像到指定路径"""
        cv2.imwrite(output_path, image)

    @staticmethod
    def save_annotations(annotations, annotation_path):
        """保存标注文件为 JSON 格式"""
        with open(annotation_path, 'w') as f:
            json.dump(annotations, f, indent=4)

    def extract_polygon(self, image, points, image_path=None, annotation_path=None):
        """
        提取多边形区域
        :param image: 原始图像
        :param points: 多边形顶点坐标
        :param image_path: 原始图像路径，用于动态命名掩膜文件
        :param annotation_path: 标注文件路径，用于动态命名掩膜文件
        :return: 提取的物体区域、掩膜、区域起始点坐标
        """
        points = np.array(points, dtype=np.int32)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)  # 掩膜与原图尺寸一致
        cv2.fillPoly(mask, [points], 255)

        object_image = cv2.bitwise_and(image, image, mask=mask)
        x, y, w, h = cv2.boundingRect(points)
        object_image = object_image[y:y + h, x:x + w]  # 物体区域大小为裁剪后的尺寸

        if image_path and annotation_path:
            # 动态生成掩膜路径
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_dir = os.path.join(self.save_dir, 'mask')
            os.makedirs(mask_dir, exist_ok=True)
            save_mask_path = os.path.join(mask_dir, f"{image_name}_mask.png")
            self.save_image(mask, save_mask_path)

        if image_path and annotation_path:
            # 动态生成掩膜路径
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            object_dir = os.path.join(self.save_dir, 'objects')
            os.makedirs(object_dir, exist_ok=True)
            save_object_path = os.path.join(object_dir, f"{image_name}.png")
            self.save_image(object_image, save_object_path)

        return object_image, mask, x, y

    @staticmethod
    def get_random_position(image, object_image):
        """
        生成随机位置，确保物体可以放置在图像内
        :param image: 原始图像
        :param object_image: 物体图像
        :return: 新的起始坐标
        """
        height, width, _ = image.shape
        obj_height, obj_width, _ = object_image.shape
        max_x = width - obj_width
        max_y = height - obj_height
        return random.randint(0, max_x), random.randint(0, max_y)

    def paste_polygon(self, image, object_image, mask, new_x, new_y, old_x, old_y):
        """
        将多边形区域粘贴到新位置
        :param image: 原始图像
        :param object_image: 提取的物体图像
        :param mask: 掩膜（原图尺寸）
        :param new_x: 新位置的起始 x 坐标
        :param new_y: 新位置的起始 y 坐标
        :param old_x: 原位置的起始 x 坐标
        :param old_y: 原位置的起始 y 坐标
        :return: 修改后的图像和新的掩膜
        """
        modified_image = image.copy()
        obj_height, obj_width, _ = object_image.shape
        mask_region = mask[old_y:old_y + obj_height, old_x:old_x + obj_width]  # 从掩膜中裁剪出物体区域

        modified_image[new_y:new_y + obj_height, new_x:new_x + obj_width] = np.where(
            mask_region[:, :, None] == 255,
            object_image,
            modified_image[new_y:new_y + obj_height, new_x:new_x + obj_width]
        )

        new_mask = np.zeros(image.shape[:2], dtype=np.uint8)  # 新掩膜与原图尺寸一致
        new_mask[new_y:new_y + obj_height, new_x:new_x + obj_width] = mask_region

        return modified_image, new_mask

    def update_annotations(self, annotations, new_x, new_y, shape_idx=0):
        """
        更新标注文件中的顶点坐标
        :param annotations: 标注数据
        :param new_x: 目标位置的起始 x 坐标
        :param new_y: 目标位置的起始 y 坐标
        :param shape_idx: 要更新的形状索引
        :return: 更新后的标注数据
        """
        # 获取当前形状的点
        points = annotations['shapes'][shape_idx]['points']
        # 计算相对偏移量
        offset_x = new_x - np.min([p[0] for p in points])
        offset_y = new_y - np.min([p[1] for p in points])

        # 更新每个点的坐标
        updated_points = []
        for x, y in points:
            updated_points.append([x + offset_x, y + offset_y])

        # 更新标注数据
        annotations['shapes'][shape_idx]['points'] = updated_points
        return annotations

    @staticmethod
    def find_annotation_file(object_image_path, annotation_dir='cropped_labels'):
        """
        根据物体图像文件名找到对应的标注文件路径
        :param object_image_path: 物体图像路径
        :param annotation_dir: 标注文件所在目录
        :return: 标注文件路径
        """
        object_name = os.path.splitext(os.path.basename(object_image_path))[0]  # 提取物体图像名称
        annotation_path = os.path.join(annotation_dir, f"{object_name}.json")
        if os.path.exists(annotation_path):
            return annotation_path
        else:
            raise FileNotFoundError(f"Annotation file for {object_name} not found in {annotation_dir}.")

    @staticmethod
    def compare_images(image1_path, image2_path):
        # 加载两张图像
        image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

        # 检查图像尺寸是否一致
        if image1.shape != image2.shape:
            print("两张图像尺寸不同，无法比较。")
            return False

        # 计算像素差值
        difference = np.abs(image1 - image2)

        # 可视化差异（可选）
        cv2.imshow("Difference", difference)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 统计差异（如平均差异）
        mean_difference = np.mean(difference)
        print(f"平均像素差值: {mean_difference}")

        # 根据阈值判断是否相同
        threshold = 10  # 可调整阈值
        return mean_difference < threshold

    def copy_paste(self, image_path, annotation_path, output_image_path, output_annotation_path, output_mask_path):
        """
        执行 Copy-Paste 操作
        :param image_path: 原始图像路径
        :param annotation_path: 标注文件路径
        :param output_image_path: 修改后图像的保存路径
        :param output_annotation_path: 修改后标注文件的保存路径
        """
        image = self.load_image(image_path)
        annotations = self.load_annotations(annotation_path)

        shape = annotations['shapes'][0]
        points = shape['points']
        object_image, mask, x, y = self.extract_polygon(image, points, image_path, annotation_path)

        new_x, new_y = self.get_random_position(image, object_image)
        modified_image, modified_mask = self.paste_polygon(image, object_image, mask, new_x, new_y, x, y)

        updated_annotations = self.update_annotations(annotations, new_x, new_y)

        self.save_image(modified_image, output_image_path)
        self.save_image(modified_mask, output_mask_path)
        self.save_annotations(updated_annotations, output_annotation_path)
        print(f"Copy-Paste 完成，结果已保存到 {output_image_path}, {output_mask_path} 和 {output_annotation_path}。")

    def splice(self, target_image_path, object_image, output_image_path, output_annotation_path, annotation_data, output_mask_path):
        """
        将物体图像拼接到目标图像中并保存掩膜和标注文件
        :param target_image_path: 目标图像路径
        :param object_image: 物体图像
        :param output_image_path: 拼接后图像保存路径
        :param output_annotation_path: 拼接后标注文件保存路径
        :param annotation_data: 物体图像对应的标注数据
        :param output_mask_path: 拼接物体在目标图像的掩膜保存路径
        """
        target_image = self.load_image(target_image_path)
        new_x, new_y = self.get_random_position(target_image, object_image)

        obj_height, obj_width, _ = object_image.shape
        max_x = min(new_x + obj_width, target_image.shape[1])
        max_y = min(new_y + obj_height, target_image.shape[0])

        modified_image = target_image.copy()
        new_mask = np.zeros(target_image.shape[:2], dtype=np.uint8)

        if max_y > new_y and max_x > new_x:
            object_region = object_image[:max_y - new_y, :max_x - new_x]
            modified_image[new_y:max_y, new_x:max_x] = np.where(
                object_region != 0,
                object_region,
                target_image[new_y:max_y, new_x:max_x]
            )

        # 更新目标图像和掩膜
        for shape in annotation_data['shapes']:
            points = np.array(shape['points'], dtype=np.int32)
            # 计算相对偏移
            offset_x = new_x - np.min(points[:, 0])
            offset_y = new_y - np.min(points[:, 1])

            # 更新点的绝对坐标
            shifted_points = points.copy()
            shifted_points[:, 0] += offset_x
            shifted_points[:, 1] += offset_y
            cv2.fillPoly(new_mask, [shifted_points], 255)  # 绘制多边形掩膜

        # 保存拼接后图像和掩膜
        self.save_image(modified_image, output_image_path)
        self.save_image(new_mask, output_mask_path)

        # 更新并保存拼接后的标注文件
        for shape in annotation_data['shapes']:
            for point in shape['points']:
                point[0] += offset_x
                point[1] += offset_y
        self.save_annotations(annotation_data, output_annotation_path)

        print(f"Splice 完成，结果已保存到 {output_image_path}, {output_mask_path} 和 {output_annotation_path}。")

    def inpaint_with_lama(self, img_path, mask_path, output_path):
        """
        使用 SimpleLama 进行图像修复
        :param img_path: 输入图像路径
        :param mask_path: 输入掩膜路径
        :param output_path: 修复后图像保存路径
        """
        simple_lama = SimpleLama()

        # 加载图像和掩膜
        image = Image.open(img_path)
        print(img_path)
        mask = Image.open(mask_path).convert('L')
        print(mask_path)
        mask_array = np.array(mask)  # 将 PIL.Image.Image 转换为 NumPy 数组
        # self.save_image(mask_array, 'output/mask/test.png')

        # 执行修复
        result = simple_lama(image, mask)

        # 保存结果
        result.save(output_path)
        print(f"伪造图像已保存到 {output_path}")



# 提供交互式菜单
def get_user_operation():
    print("请选择操作类型：")
    print("1. 执行 Copy-Paste 操作")
    print("2. 执行 Spliced 操作")
    print("3. 执行 Inpaint 操作")
    operation = input("请输入数字 1 , 2 或 3（默认为 1）: ").strip()
    return operation or "1"


def main():
    processor = ImageProcessor()

    # input_image_dir = 'images'  # 输入的图像路径
    # input_annotation_dir = 'annotations'  # 输入的标注1文件路径
    input_image_dir = 'cropped_images'  # 输入的图像路径
    input_annotation_dir = 'cropped_labels'  # 输入的标注文件路径

    output_copy_paste_image_dir = 'output/modified_images/copy_paste'  # copy-paste后图像保存路径
    output_copy_paste_mask_dir = 'output/modified_masks/copy_paste'  # copy-paste后图像mask保存路径
    output_copy_paste_annotation_dir = 'output/modified_annotations/copy_paste'  # copy-paste后标注保存路径
    output_splice_image_dir = 'output/modified_images/spliced'  # 拼接图像保存路径
    output_splice_mask_dir = 'output/modified_masks/spliced'  # 拼接图像mask保存路径
    output_splice_annotation_dir = 'output/modified_annotations/spliced'  # 拼接标注文件保存路径
    output_inpainted_dir = 'output/modified_images/inpainted' # inpainting后的图像保存路径

    # 创建输出目录
    os.makedirs(output_copy_paste_image_dir, exist_ok=True)
    os.makedirs(output_copy_paste_mask_dir, exist_ok=True)
    os.makedirs(output_copy_paste_annotation_dir, exist_ok=True)
    os.makedirs(output_splice_image_dir, exist_ok=True)
    os.makedirs(output_splice_mask_dir, exist_ok=True)
    os.makedirs(output_splice_annotation_dir, exist_ok=True)
    os.makedirs(output_inpainted_dir, exist_ok=True)

    if len(sys.argv) < 2:
        operation = get_user_operation()
    else:
        operation = sys.argv[1]


    if operation == "1":
        print("执行 Copy-Paste 操作")
        # 获取文件列表
        all_images = [f for f in os.listdir(input_image_dir) if f.endswith('.jpg')]

        for file_name in all_images:
            base_name = os.path.splitext(file_name)[0]

            # 输入文件路径
            image_path = os.path.join(input_image_dir, file_name)
            annotation_path = os.path.join(input_annotation_dir, f"{base_name}.json")

            # 输出文件路径
            base_name = os.path.splitext(file_name)[0]
            output_copy_paste_image_path = os.path.normpath(os.path.join(output_copy_paste_image_dir, f"{base_name}.jpg"))
            output_copy_paste_mask_path = os.path.normpath(os.path.join(output_copy_paste_mask_dir, f"{base_name}.png"))
            output_copy_paste_annotation_path = os.path.normpath(os.path.join(output_copy_paste_annotation_dir, f"{base_name}.json"))

            print(f"Processing copy-paste for: {file_name}")
            processor.copy_paste(image_path, annotation_path, output_copy_paste_image_path,
                                 output_copy_paste_annotation_path, output_copy_paste_mask_path)

    elif operation == "2":
        print("执行 Spliced 操作")
        # 获取文件列表
        all_images = [f for f in os.listdir(input_image_dir) if f.endswith('.jpg')]

        for target_image in all_images:
            target_base_name = os.path.splitext(target_image)[0]

            # 当前 target_image 的路径
            spliced_target_image_path = os.path.join(input_image_dir, target_image)

            # 随机选择对象图像，确保与目标图像不同
            filtered_images = [
                img for img in all_images if os.path.splitext(img)[0] != target_base_name
            ]

            if not filtered_images:
                print(f"没有足够的对象图像与目标图像 {target_image} 进行拼接操作。")
                continue

            object_image_file = random.choice(filtered_images)
            # object_image_path = os.path.join('output/objects', object_image_file)
            # object_image_path = Path('output') / 'objects' / object_image_file

            # 1. 分离出不带后缀的文件名
            base_name = os.path.splitext(object_image_file)[0]  # 如果 object_image_file = "xxx.jpg"，则 base_name = "xxx"

            # 2. 拼接新的后缀 .png
            object_image_file_png = base_name + ".png"  # "xxx.png"
            # object_image_file_jpg = base_name + ".jpg"  # "xxx.jpg"

            # 3. 构造完整路径（示例：原来放在 'output/objects' 目录下）
            object_image_path = Path('output') / 'objects' / object_image_file_png
            # object_image_path = Path('output') / 'objects' / object_image_file_jpg

            if not object_image_path.exists():
                print(f"错误：文件不存在 - {object_image_path}")
            else:
                # 读取图片
                object_image = cv2.imread(str(object_image_path))
                if object_image is None:
                    print(f"错误：无法读取图片 - {object_image_path}")
                else:
                    print("图片读取成功！")

            object_base_name = os.path.splitext(object_image_file)[0]

            print(f"Processing splice for: {object_image_file_png} to {target_image}")

            # 对象图像的标注文件路径
            annotation_file_path = processor.find_annotation_file(object_image_path)
            annotation_data = processor.load_annotations(annotation_file_path)
            object_image = processor.load_image(object_image_path)

            # 输出拼接后的文件路径
            output_splice_image_path = os.path.normpath(os.path.join(output_splice_image_dir, f"{object_base_name}_to_{target_base_name}.jpg"))
            output_splice_mask_path = os.path.normpath(os.path.join(output_splice_mask_dir, f"{object_base_name}_to_{target_base_name}.png"))
            output_splice_annotation_path = os.path.normpath(os.path.join(output_splice_annotation_dir, f"{object_base_name}_to_{target_base_name}.json"))

            processor.splice(spliced_target_image_path, object_image, output_splice_image_path,
                             output_splice_annotation_path, annotation_data, output_splice_mask_path)

    elif operation == "3":
        print("执行 Inpainting 篡改操作")

        # 1. 找到 images 文件夹下所有 jpg 文件
        all_images = [f for f in os.listdir(input_image_dir) if f.endswith('.jpg')]
        if not all_images:
            print("images 文件夹中没有找到任何 JPG 文件。")
            sys.exit(0)

        for jpg_file in all_images:
            base_name = os.path.splitext(jpg_file)[0]

            # 2. 推断 mask 文件名，一般为 “<base_name>_mask.png” 或者别的规则
            #    假设你在 extract_polygon() 保存 mask 时，文件名是 "<image_name>_mask.png"
            mask_path = os.path.join('output', 'mask', f"{base_name}_mask.png")
            #    如果你的 mask 命名不带 _mask ，只是同名 png，可改成:
            #    mask_path = os.path.join('output', 'mask', f"{base_name}.png")

            # 3. 拼接原图完整路径
            image_path = os.path.join(input_image_dir, jpg_file)

            # -- 新增：检查通道数，如超过3通道就转为3通道 --

            img_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                print(f"无法读取图像: {image_path}")
                continue

            # img_bgr.shape 通常是 (height, width, channels)，若 channels>3 则说明带额外通道
            if len(img_bgr.shape) == 3 and img_bgr.shape[2] > 3:
                # 用 PIL 把它转换为RGB三通道
                with Image.open(image_path) as im:
                    im_rgb = im.convert("RGB")
                    # 重新保存到一个新文件，或直接覆盖也行（此处示例写入新文件）
                    new_jpg_path = os.path.join(input_image_dir, f"{base_name}_rgb.jpg")
                    im_rgb.save(new_jpg_path, "JPEG")
                    print(f"注意: {jpg_file} 检测到超过3通道，已转换为RGB三通道 => {new_jpg_path}")
                    image_path = new_jpg_path

            # 4. 判断 mask 文件是否存在
            if not os.path.exists(mask_path):
                print(f"未找到对应的掩膜文件: {mask_path}，跳过...")
                continue

            # 5. 构造输出文件路径
            output_inpainted_image_path = os.path.join(
                output_inpainted_dir,
                f"{base_name}_inpainted.jpg"  # 最终保存为 JPG
            )

            print(f"开始处理: {image_path}，使用 mask: {mask_path}")
            processor.inpaint_with_lama(image_path, mask_path, output_inpainted_image_path)
            print(f"Inpainting 完成: {output_inpainted_image_path}")

    else:
        print("无效的操作类型，请输入 1, 2 或 3")
        sys.exit(1)



if __name__ == "__main__":
    main()
