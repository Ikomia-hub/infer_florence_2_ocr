<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_florence_2_ocr</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_florence_2_ocr">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_florence_2_ocr">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_florence_2_ocr/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_florence_2_ocr.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks. 
In this algorithm you can use Florence-2 for text recognition (OCR). 

![ocr illustration](https://raw.githubusercontent.com/Ikomia-hub/infer_florence_2_ocr/main/images/output.jpg)



## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_florence_2_ocr", auto_connect=True)

# Run on your image  
wf.run_on(url="https://images.pexels.com/photos/5248077/pexels-photo-5248077.jpeg?cs=srgb&dl=pexels-leeloothefirst-5248077.jpg&fm=jpg&w=640&h=960")

# Display results
img_output = algo.get_output(0)
recognition_output = algo.get_output(1)
display(img_output.get_image_with_mask_and_graphics(recognition_output), title="Florence-2 OCR")
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.
- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters
- **model_name** (str) - default 'microsoft/Florence-2-base': Name of the Florence-2 pre-trained model. Other models available:
    - microsoft/Florence-2-large
    - microsoft/Florence-2-base-ft
    - microsoft/Florence-2-large-ft
- **num_beams** (int) - default '3': By specifying a number of beams higher than 1, you are effectively switching from greedy search to beam search. This strategy evaluates several hypotheses at each time step and eventually chooses the hypothesis that has the overall highest probability for the entire sequence. This has the advantage of identifying high-probability sequences that start with a lower probability initial tokens and would’ve been ignored by the greedy search. 
- **do_sample** (bool) - default 'False': If set to True, this parameter enables decoding strategies such as multinomial sampling, beam-search multinomial sampling, Top-K sampling and Top-p sampling. All these strategies select the next token from the probability distribution over the entire vocabulary with various strategy-specific adjustments.
- **early_stopping** (bool) - default 'False': Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values: True, where the generation stops as soon as there are num_beams complete candidates; False, where an heuristic is applied and the generation stops when is it very unlikely to find better candidates; "never", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).
- **cuda** (bool): If True, CUDA-based inference (GPU). If False, run on CPU.
Optionally, you can load a custom model: 


**Parameters** should be in **strings format**  when added to the dictionary.

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_florence_2_ocr", auto_connect=True)

algo.set_parameters({
    "model_name":"microsoft/Florence-2-large",
    "max_new_tokens":"1024",
    "num_beams":"3",
    "do_sample":"False",
    "early_stopping":"False",
    "cuda":"True"
})

wf.run_on(url="https://images.pexels.com/photos/5248077/pexels-photo-5248077.jpeg?cs=srgb&dl=pexels-leeloothefirst-5248077.jpg&fm=jpg&w=640&h=960")

# Display results
img_output = algo.get_output(0)
recognition_output = algo.get_output(1)
display(img_output.get_image_with_mask_and_graphics(recognition_output), title="Florence-2 OCR")
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_florence_2_ocr", auto_connect=True)

# Run on your image  
wf.run_on(url="https://images.pexels.com/photos/5248077/pexels-photo-5248077.jpeg?cs=srgb&dl=pexels-leeloothefirst-5248077.jpg&fm=jpg&w=640&h=960")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```
