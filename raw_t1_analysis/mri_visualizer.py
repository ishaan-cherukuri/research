import matplotlib.pyplot as plt
import dask.array as da
import numpy as np
import sys

from vtkmodules.vtkRenderingCore import (
    vtkRenderWindow,
    vtkRenderer,
    vtkRenderWindowInteractor,
    vtkTextActor,
)
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkCommonCore import vtkUnsignedCharArray
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
from vtkmodules.vtkRenderingCore import vtkVolume
from vtkmodules.vtkRenderingCore import vtkVolumeProperty
from vtkmodules.util import numpy_support
from vtkmodules.vtkInteractionWidgets import vtkSliderWidget, vtkSliderRepresentation2D


def create_slider_widget(
    interactor,
    min_value,
    max_value,
    value,
    title,
    callback,
    x1=0.1,
    y1=0.05,
    x2=0.4,
    y2=0.1,
):
    slider_rep = vtkSliderRepresentation2D()
    slider_rep.SetMinimumValue(min_value)
    slider_rep.SetMaximumValue(max_value)
    slider_rep.SetValue(value)
    slider_rep.SetTitleText(title)
    slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint1Coordinate().SetValue(x1, y1)
    slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint2Coordinate().SetValue(x2, y2)
    slider_rep.SetSliderLength(0.02)
    slider_rep.SetSliderWidth(0.03)
    slider_rep.SetEndCapLength(0.01)
    slider_rep.SetEndCapWidth(0.03)
    slider_rep.SetTubeWidth(0.005)
    slider_rep.SetLabelFormat("%0.0f")
    slider_rep.SetTitleHeight(0.02)
    slider_rep.SetLabelHeight(0.02)

    slider_widget = vtkSliderWidget()
    slider_widget.SetInteractor(interactor)
    slider_widget.SetRepresentation(slider_rep)
    slider_widget.SetAnimationModeToAnimate()
    slider_widget.EnabledOn()
    slider_widget.AddObserver(16, callback)  # 16 = vtkCommand.InteractionEvent
    return slider_widget


def visualize_mri_3d(image_path):
    """
    Comprehensive 3D MRI visualization with VTK, including window/level, opacity, and slice controls.
    """

    def show_2d_slices(arr):
        """Show axial, sagittal, and coronal slices in a matplotlib window."""
        mid_z = arr.shape[2] // 2
        mid_y = arr.shape[1] // 2
        mid_x = arr.shape[0] // 2
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(arr[:, :, mid_z], cmap="gray")
        axes[0].set_title(f"Axial (Z={mid_z})")
        axes[1].imshow(arr[:, mid_y, :], cmap="gray")
        axes[1].set_title(f"Coronal (Y={mid_y})")
        axes[2].imshow(arr[mid_x, :, :], cmap="gray")
        axes[2].set_title(f"Sagittal (X={mid_x})")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    # Load MRI image using nibabel and dask
    from nibabel.loadsave import load

    img = load(image_path)
    data = img.get_fdata()  # type: ignore
    dask_data = da.from_array(data, chunks=[64, 64, 64])
    arr = dask_data.compute()
    arr = np.nan_to_num(arr)
    arr = (255 * (arr - arr.min()) / (arr.max() - arr.min())).astype(np.uint8)

    # Show 2D slices in a separate window
    show_2d_slices(arr)

    # Convert numpy array to VTK image
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=arr.ravel(order="F"),
        deep=True,
        array_type=vtkUnsignedCharArray().GetDataType(),
    )
    vtk_img = vtkImageData()
    vtk_img.SetDimensions(arr.shape)
    vtk_img.SetSpacing(tuple(float(x) for x in img.header.get_zooms()))  # type: ignore
    vtk_img.GetPointData().SetScalars(vtk_data)

    # Set up VTK volume rendering
    from vtkmodules.vtkRenderingCore import vtkColorTransferFunction
    from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

    volume_mapper = vtkSmartVolumeMapper()
    volume_mapper.SetInputData(vtk_img)

    # Color transfer function (grayscale)
    color_func = vtkColorTransferFunction()
    color_func.AddRGBPoint(0, 0.0, 0.0, 0.0)
    color_func.AddRGBPoint(255, 1.0, 1.0, 1.0)

    # Opacity transfer function
    opacity_func = vtkPiecewiseFunction()
    opacity_func.AddPoint(0, 0.0)
    opacity_func.AddPoint(10, 0.0)
    opacity_func.AddPoint(40, 0.05)
    opacity_func.AddPoint(80, 0.15)
    opacity_func.AddPoint(120, 0.3)
    opacity_func.AddPoint(180, 0.5)
    opacity_func.AddPoint(255, 0.7)

    volume_property = vtkVolumeProperty()
    volume_property.SetColor(color_func)
    volume_property.SetScalarOpacity(opacity_func)
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()

    volume = vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    # Set up renderer and window
    renderer = vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.2)
    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1000, 900)
    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    style = vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)

    # Add text actor for slice info
    text_actor = vtkTextActor()
    text_actor.SetInput(f"Shape: {arr.shape}")
    text_actor.GetTextProperty().SetFontSize(18)
    text_actor.GetTextProperty().SetColor(1, 1, 1)
    text_actor.SetPosition(10, 10)
    renderer.AddActor2D(text_actor)

    # Opacity slider callback
    def opacity_callback(obj, event):
        value = obj.GetRepresentation().GetValue()
        volume_property.SetScalarOpacityUnitDistance(value)
        render_window.Render()

    # Window/level slider callback
    def window_callback(obj, event):
        value = obj.GetRepresentation().GetValue()
        volume_property.SetColor([0, value / 255.0, 1 - value / 255.0])
        render_window.Render()

    # Slice slider callback (shows a single slice as a 2D image)
    slice_actor = None

    def slice_callback(obj, event):
        nonlocal slice_actor
        idx = int(obj.GetRepresentation().GetValue())
        if slice_actor:
            renderer.RemoveActor(slice_actor)
        from vtkmodules.vtkRenderingCore import vtkImageActor

        slice_data = arr[:, :, idx]
        slice_vtk = numpy_support.numpy_to_vtk(
            num_array=slice_data.ravel(order="F"),
            deep=True,
            array_type=vtkUnsignedCharArray().GetDataType(),
        )
        slice_img = vtkImageData()
        slice_img.SetDimensions(slice_data.shape[0], slice_data.shape[1], 1)
        slice_img.GetPointData().SetScalars(slice_vtk)
        slice_actor = vtkImageActor()
        slice_actor.SetInputData(slice_img)
        renderer.AddActor(slice_actor)
        render_window.Render()

    # Add sliders
    create_slider_widget(
        interactor,
        0.01,
        2.0,
        0.5,
        "Opacity",
        opacity_callback,
        x1=0.1,
        y1=0.01,
        x2=0.4,
        y2=0.05,
    )
    create_slider_widget(
        interactor,
        0,
        255,
        128,
        "Window/Level",
        window_callback,
        x1=0.5,
        y1=0.01,
        x2=0.8,
        y2=0.05,
    )
    create_slider_widget(
        interactor,
        0,
        arr.shape[2] - 1,
        arr.shape[2] // 2,
        "Slice (Z)",
        slice_callback,
        x1=0.1,
        y1=0.93,
        x2=0.4,
        y2=0.97,
    )

    # Start interaction
    render_window.Render()
    interactor.Start()


# Example usage:
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mri_visualizer.py <path_to_nifti_image>")
    else:
        visualize_mri_3d(sys.argv[1])
