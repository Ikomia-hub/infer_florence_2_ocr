import copy
import torch
import os
import numpy as np
from ikomia import core, dataprocess, utils
from transformers import AutoProcessor, AutoModelForCausalLM


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferFlorence2OcrParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.model_name = 'microsoft/Florence-2-large'
        self.max_new_tokens = 1024
        self.num_beams = 3
        self.do_sample = False
        self.early_stopping = False
        self.cuda = torch.cuda.is_available()
        self.update = False

    def set_values(self, params):
        # Set parameters values from Ikomia Studio or API
        self.update = utils.strtobool(params["cuda"]) != self.cuda or self.model_name != str(params["model_name"])
        self.model_name = str(params["model_name"])
        self.max_new_tokens = int(params["max_new_tokens"])
        self.num_beams = int(params["num_beams"])
        self.do_sample = utils.strtobool(params["do_sample"])
        self.early_stopping = utils.strtobool(params["early_stopping"])
        self.cuda = utils.strtobool(params["cuda"])

    def get_values(self):
        # Send parameters values to Ikomia Studio or API
        # Create the specific dict structure (string container)
        params = {}
        params["model_name"] = str(self.model_name)
        params["max_new_tokens"] = str(self.max_new_tokens)
        params["num_beams"] = str(self.num_beams)
        params["do_sample"] = str(self.do_sample)
        params["early_stopping"] = str(self.early_stopping)
        params["cuda"] = str(self.cuda)

        return params

# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferFlorence2Ocr(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        self.add_output(dataprocess.CTextIO())
        self.add_output(dataprocess.DataDictIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferFlorence2OcrParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.processor = None
        self.model = None
        self.model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")
        self.device = torch.device("cpu")
        self.task_prompt = '<OCR_WITH_REGION>'
        self.color = [255, 0, 0]

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def load_model(self, param):
        try:
            self.processor = AutoProcessor.from_pretrained(
                                    param.model_name,
                                    cache_dir=self.model_folder,
                                    local_files_only=True,
                                    trust_remote_code=True
                                    )

            self.model = AutoModelForCausalLM.from_pretrained(
                                    param.model_name,
                                    cache_dir=self.model_folder,
                                    local_files_only=True,
                                    trust_remote_code=True
                                    ).eval()

        except Exception as e:
            print(f"Failed with error: {e}. Trying without the local_files_only parameter...")
            self.processor = AutoProcessor.from_pretrained(
                                        param.model_name,
                                        cache_dir=self.model_folder,
                                        trust_remote_code=True
                                        )

            self.model = AutoModelForCausalLM.from_pretrained(
                                    param.model_name,
                                    cache_dir=self.model_folder,
                                    trust_remote_code=True
                                    ).eval()
        self.model.to(self.device)

    def infer(self, img, param):
        # Image pre-process
        img_h, img_w = img.shape[:2]
        inputs = self.processor(
                        text=self.task_prompt,
                        images=img,
                        return_tensors="pt"
                    ).to(self.device)

        # Inference
        generated_ids = self.model.generate(
                                    input_ids=inputs["input_ids"],
                                    pixel_values=inputs["pixel_values"],
                                    max_new_tokens=param.max_new_tokens,
                                    early_stopping=param.early_stopping,
                                    do_sample=param.do_sample,
                                    num_beams=param.num_beams,
                                    )
        generated_text = self.processor.batch_decode(
                                            generated_ids,
                                            skip_special_tokens=False
                                            )[0]
        parsed_answer = self.processor.post_process_generation(
                                            generated_text,
                                            task=self.task_prompt,
                                            image_size=(img_w, img_h)
                                            )

        return parsed_answer

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters
        param = self.get_param_object()

        # Get input :
        input = self.get_input(0)

        # Get output :
        text_output = self.get_output(1)
        output_dict = self.get_output(2)
        self.forward_input_image(0, 0)

        # Get image from input/output (numpy array):
        src_image = input.get_image()

        # Load model
        if param.update or self.model is None:
            self.device = torch.device(
                "cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
            self.load_model(param)
            param.update = False

        # Inference
        with torch.no_grad():
            results = self.infer(src_image, param)

        # Set output dict
        output_dict.data = results

        prediction = results['<OCR_WITH_REGION>']
        bboxes, labels = prediction['quad_boxes'], prediction['labels']

        # Edit the first label
        if len(labels)>0 and '</s>' in labels[0]:
            labels[0] = labels[0].replace('</s>', '')

        for i, (text_box, label) in enumerate(zip(bboxes, labels)):
            # text_box contains 8 values: [x1, y1, x2, y2, x3, y3, x4, y4]
            # Convert to numpy array for easier manipulation
            box = np.array(text_box).reshape(-1, 2)  # Reshape to (4, 2) for four points

            # Calculate bounding box: take the min and max coordinates
            x_min, y_min = np.min(box, axis=0)
            x_max, y_max = np.max(box, axis=0)

            # Calculate width and height
            w = x_max - x_min
            h = y_max - y_min

            # Add text graphics object
            text_output.add_text_field(
                            id=i,
                            label="",
                            text=label,
                            confidence=1,
                            box_x=x_min,
                            box_y=y_min,
                            box_width=w,
                            box_height=h,
                            color=self.color
            )

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()

# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferFlorence2OcrFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_florence_2_ocr"
        self.info.short_description = "Inference for text recognition (OCR) with Florence-2"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/OCR"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "B. Xiao, H. Wu, W. Xu, X. Dai, H. Hu, Y. Lu, M. Zeng, C. Liu, L. Yuan"
        self.info.article = "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks"
        self.info.journal = "arXiv:2311.06242"
        self.info.year = 2023
        self.info.license = "MIT License"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_florence_2_caption"
        self.info.original_repository = "https://github.com/googleapis/python-vision"
        # Python version
        self.info.min_python_version = "3.10.0"
        # Keywords used for search
        self.info.keywords = "Florence,Microsoft,Captioning,Unified,Pytorch"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OCR"
        self.info.os = utils.OSType.LINUX

    def create(self, param=None):
        # Create algorithm object
        return InferFlorence2Ocr(self.info.name, param)
