import functools
import tempfile
from typing import Any
from typing import Tuple, List, Optional

import gradio
import gradio as gr
import matplotlib.pyplot as plt
import torch

from dust3r.inference import load_model
from func import (
    string_to_port_crc32,
    get_reconstructed_scene,
    get_3d_model_from_scene,
)

plt.ion()
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12


# Constants
LOCAL_CLIENT_IP: str = "0.0.0.0"
APP_NAME: str = "easy3d_reconstructor"
DEFAULT_PORT: int = string_to_port_crc32(APP_NAME)  # 28439
BATCH_SIZE = 1
IMAGE_SIZE = 512  # choices might include 512, 224


def setup_gradio_interface(
    predicament: Any,  # Specify the exact type here based on your application context
    model: Any,  # Specify the exact type here based on your application context
    device: Any,  # Specify the exact type here based on your application context
    image_size: int,
) -> gr.Blocks:
    """
    Set up the Gradio interface for the 3D reconstruction application.

    :param predicament: The output directory or other context-specific parameter.
    :param model: The model used for 3D reconstruction.
    :param device: The device (CPU/GPU) used for processing.
    :param image_size: The image size used for processing.
    :return: Configured Gradio interface.
    """
    # Partial functions for reconstruction and model generation
    recon_fun = functools.partial(
        get_reconstructed_scene, predicament, model, device, image_size
    )
    model_from_scene_fun = functools.partial(get_3d_model_from_scene, predicament)

    with gr.Blocks(
        css=".gradio-container {margin: 0 !important; min-width: 100%};",
        title="Easy3D Reconstructor",
    ) as interface:
        scene = gr.State(None)

        gr.Markdown(
            "# [Easy3D Reconstructor](https://github.com/CyFeng16/easy3d_reconstructor)"
        )

        with gr.Column():
            inputfiles = gr.File(file_count="multiple")

            with gr.Row():
                schedule = gr.Dropdown(
                    ["linear", "cosine"],
                    value="linear",
                    label="Schedule",
                    info="For global alignment!",
                )
                niter = gr.Number(
                    value=300,
                    precision=0,
                    minimum=0,
                    maximum=5000,
                    label="Num Iterations",
                    info="For global alignment!",
                )
                scenegraph_type = gr.Dropdown(
                    ["complete", "swin", "oneref"],
                    value="complete",
                    label="Scenegraph",
                    info="Define how to make pairs",
                    interactive=True,
                )
                winsize, refid = gr.Slider(visible=False), gr.Slider(visible=False)

            run_btn = gr.Button("Run")

            with gr.Row():
                min_conf_thr = gr.Slider(
                    label="Min Conf Thr", value=3.0, minimum=1.0, maximum=20, step=0.1
                )
                cam_size = gr.Slider(
                    label="Cam Size", value=0.05, minimum=0.001, maximum=0.1, step=0.001
                )

            with gr.Row():
                as_pointcloud = gr.Checkbox(value=False, label="As Pointcloud")
                mask_sky = gr.Checkbox(value=False, label="Mask Sky")
                clean_depth = gr.Checkbox(value=True, label="Clean-up Depthmaps")
                transparent_cams = gr.Checkbox(value=False, label="Transparent Cameras")

            outmodel, outgallery = gr.Model3D(), gr.Gallery(
                label="RGB, Depth, Confidence", columns=3, height="100%"
            )

            # Event bindings for dynamic UI interactions
            scenegraph_type.change(
                set_scenegraph_options,
                inputs=[inputfiles, winsize, refid, scenegraph_type],
                outputs=[winsize, refid],
            )
            inputfiles.change(
                set_scenegraph_options,
                inputs=[inputfiles, winsize, refid, scenegraph_type],
                outputs=[winsize, refid],
            )

            run_btn.click(
                fn=recon_fun,
                inputs=[
                    inputfiles,
                    schedule,
                    niter,
                    min_conf_thr,
                    as_pointcloud,
                    mask_sky,
                    clean_depth,
                    transparent_cams,
                    cam_size,
                    scenegraph_type,
                    winsize,
                    refid,
                ],
                outputs=[scene, outmodel, outgallery],
            )

            # Updating the model based on user adjustments without re-running the inference
            for input_widget in [
                min_conf_thr,
                cam_size,
                as_pointcloud,
                mask_sky,
                clean_depth,
                transparent_cams,
            ]:
                input_widget.change(
                    model_from_scene_fun,
                    inputs=[
                        scene,
                        min_conf_thr,
                        as_pointcloud,
                        mask_sky,
                        clean_depth,
                        transparent_cams,
                        cam_size,
                    ],
                    outputs=[outmodel],
                )

    return interface


def set_scenegraph_options(
    inputfiles: Optional[List[str]], winsize: int, refid: int, scenegraph_type: str
) -> Tuple[gradio.Slider, gradio.Slider]:
    """
    Configures and returns Gradio sliders for setting the scene graph window size and reference ID.

    :param inputfiles: The list of input files.
    :param winsize: Initial value for the window size slider.
    :param refid: Initial value for the reference ID slider.
    :param scenegraph_type: The type of scene graph, affecting which sliders are visible.
    :return: A tuple containing the window size and reference ID sliders.
    """
    num_files = len(inputfiles) if inputfiles else 1
    max_winsize = max(1, (num_files - 1) // 2)

    winsize_slider = gradio.Slider(
        label="Scene Graph: Window Size",
        value=min(winsize, max_winsize),
        minimum=1,
        maximum=max_winsize,
        step=1,
        visible=scenegraph_type == "swin",
    )

    refid_slider = gradio.Slider(
        label="Scene Graph: Id",
        value=min(refid, num_files - 1),
        minimum=0,
        maximum=num_files - 1,
        step=1,
        visible=scenegraph_type == "oneref",
    )

    return winsize_slider, refid_slider


if __name__ == "__main__":
    dust3r_model = load_model(
        "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        "cuda",
    )
    with tempfile.TemporaryDirectory(suffix="demo23d") as _predicament:
        print("Output stuff in", _predicament)
        demo = setup_gradio_interface(
            _predicament,
            dust3r_model,
            "cuda",
            IMAGE_SIZE,
        )
        demo.queue()
        demo.launch(server_name=LOCAL_CLIENT_IP, server_port=DEFAULT_PORT)
