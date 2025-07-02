from pathlib import Path
from typing import Any


from pathlib import Path
from typing import Any, Dict

from .base_pipeline import BasePipeline
from ..processor import ImageComparison


class DiffPipeline(BasePipeline):
    """
    重构后的DiffPipeline，基于BasePipeline
    """

    def __init__(self, config: dict, out_dir="output", is_debug: bool = False, max_num: int = None):
        super().__init__(
            config=config, out_dir=out_dir, is_debug=is_debug, max_num=max_num, pipeline_name="DiffPipeline", image_size=(256, 256)
        )
        # 初始化图像对比处理器
        self.image_consultant = ImageComparison(config)

    def pipeline(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        核心处理逻辑：图像差异分析
        """
        image_path = info.get("image_path", None)

        if image_path:
            # 处理JSON文件中的数据
            image_path = Path(image_path)
            ret = self._load_json_data(image_path)

            # 加载真实图像
            _, trans_real_img, _ = self.image_processor.load_image(ret["real_image"])
            real_image_base64 = self.image_processor.get_base64(trans_real_img)

            # 处理每个创意编辑结果
            for creativity_info in ret["creativity"]:
                _, trans_edited_img, _ = self.image_processor.load_image(creativity_info["output_img"])
                edited_image_base64 = self.image_processor.get_base64(trans_edited_img)

                _, trans_gt_mask_img, _ = self.image_processor.load_image(creativity_info["gt_mask"])
                gt_mask_base64 = self.image_processor.get_base64(trans_gt_mask_img)

                res_info = self.image_consultant.run(
                    image_path=ret["real_image"],
                    image_base64=real_image_base64,
                    edited_image_base64=edited_image_base64,
                    gt_mask_base64=gt_mask_base64,
                )
                creativity_info["diff"] = res_info

            return ret
        else:
            # 处理单个图像对比
            _, trans_real_img, _ = self.image_processor.load_image(info["real_image"])
            real_image_base64 = self.image_processor.get_base64(trans_real_img)

            _, trans_edited_img, _ = self.image_processor.load_image(info["fake_image"])
            edited_image_base64 = self.image_processor.get_base64(trans_edited_img)

            _, trans_gt_mask_img, _ = self.image_processor.load_image(info["gt_mask"])
            gt_mask_base64 = self.image_processor.get_base64(trans_gt_mask_img)

            res_info = self.image_consultant.run(
                image_path=info["real_image"],
                image_base64=real_image_base64,
                edited_image_base64=edited_image_base64,
                gt_mask_base64=gt_mask_base64,
            )
            info["diff"] = res_info
            return info
