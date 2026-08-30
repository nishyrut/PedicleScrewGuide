# Imports
import SimpleITK as sitk
import os
import cv2
import numpy as np
import math
import sys
import vtk
import shutil
from pathlib import Path
from vtk.util import numpy_support
from totalsegmentator.python_api import totalsegmentator

# Starting
print("Screw Guide Generator Starting")

# Variable
Testing = False
Debug = False
Run_Resampler = True
Run_Totalsegmentator = True
Single_Vertebrae_Dict = {}
Slice_MM = 0.75 # NOTE: CONSTANT
STL_Conversion = True

# Vertebrae
Spine_Vertebraes = ['vertebrae_C1', 'vertebrae_C2', 'vertebrae_C3', 'vertebrae_C4', 'vertebrae_C5', 'vertebrae_C6', 'vertebrae_C7' , 'vertebrae_T1', 'vertebrae_T2', 'vertebrae_T3', 'vertebrae_T4', 'vertebrae_T5', 'vertebrae_T6', 'vertebrae_T7', 'vertebrae_T8', 'vertebrae_T9', 'vertebrae_T10', 'vertebrae_T11', 'vertebrae_T12','vertebrae_L1', 'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5', 'sacrum']

# ---- Helper Functions ----
# Find the largest 3D object - Useful for removing noise/speckles
def largest_object(mask):
    label_image = sitk.ConnectedComponent(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(label_image)
    label_ids = stats.GetLabels()
    largest_label = max(label_ids, key=lambda l: stats.GetNumberOfPixels(l))
    largest_mask = label_image == largest_label
    return largest_mask

# Bonuding box for 3D volumes
def calc_bbox(volume):
    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(volume)
    return shape_filter.GetBoundingBox(1)

# Count the number of 3D objects
def count_3d_objects(sub_volume, threshold_value):
    spacing = sub_volume.GetSpacing()
    voxel_volume = voxel_volume = spacing[0] * spacing[1] * spacing[2]
    min_physical_size = 64
    min_size = int(min_physical_size // voxel_volume)
    binary_volume = sitk.BinaryThreshold(sub_volume, lowerThreshold=threshold_value)
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled_volume = cc_filter.Execute(binary_volume)
    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_filter.SetMinimumObjectSize(min_size)
    filtered_volume = relabel_filter.Execute(labeled_volume)
    object_count = relabel_filter.GetNumberOfObjects()
    
    return object_count

# Obtain the 2D slice as an image
def get_2D_slice(volume, slice, axis = 1):
    if axis == 0: curr_slice = volume[slice, :, :]
    if axis == 1: curr_slice = volume[:, slice, :]
    if axis == 2: curr_slice = volume[:, :, slice]
    slice_2d = sitk.GetArrayFromImage(curr_slice)
    cv_image = (slice_2d * 255).astype(np.uint8)
    return cv_image

# Resample CT volume
def resample_ct_volume(input_volume, target_spacing_mm = 1):
    target_spacing = [float(target_spacing_mm)] * 3
    original_spacing = input_volume.GetSpacing()
    original_size = input_volume.GetSize()
    target_size = [
        int(round(sz * spc / target_spacing_mm))
        for sz, spc in zip(original_size, original_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(input_volume.GetOrigin())
    resampler.SetOutputDirection(input_volume.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)
    
    return resampler.Execute(input_volume)

# Rotate volume with physical center and the angle in degrees
def rotate_volume(image, angles_deg, center_mm):
    angles_rad = [np.deg2rad(a) for a in angles_deg]
    tx = sitk.Euler3DTransform()
    tx.SetCenter(center_mm)
    tx.SetRotation(angles_rad[1], angles_rad[2], angles_rad[0])
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image) # Keeps original spacing, origin, direction
    resampler.SetInterpolator(sitk.sitkNearestNeighbor) # Best for binary masks
    resampler.SetTransform(tx)
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(image)

# Generate a 3D line
def generate_3d_line_by_length(start, stop, target_length):
    start = np.array(start, dtype=float)
    stop = np.array(stop, dtype=float)
    direction = stop - start
    distance = np.linalg.norm(direction)
    if distance == 0:
        direction = np.array([0.0, 0.0, 1.0])
    else:
        direction = direction / distance
    actual_end_point = start + (direction * target_length)
    num_points = target_length * 2
    line_points = np.linspace(start, actual_end_point, num_points)
    
    return line_points

# Dilate the generated trajectory volume
def generate_trajectory_volume(reference_mask, safe_points, kernel_radius = 1):
    blank_array = np.zeros_like(sitk.GetArrayFromImage(reference_mask))

    z_max, y_max, x_max = blank_array.shape
    for pt in safe_points:
        x, y, z = round(pt[0]), round(pt[1]), round(pt[2])
        if 0 <= z < z_max and 0 <= y < y_max and 0 <= x < x_max:
            blank_array[z, y, x] = 1

    path_image = sitk.GetImageFromArray(blank_array)
    path_image.CopyInformation(reference_mask)

    dilator = sitk.BinaryDilateImageFilter()
    dilator.SetKernelType(sitk.sitkBall)
    dilator.SetKernelRadius(kernel_radius)
    dilator.SetForegroundValue(1)

    final_volume = dilator.Execute(path_image)

    return final_volume

# Export to STL after smoothing the volume
def sitk_to_stl(sitk_image, output_path, iterations=50):
    img_array = sitk.GetArrayFromImage(sitk_image)
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(sitk_image.GetSize())
    vtk_img.SetSpacing(sitk_image.GetSpacing())
    vtk_img.SetOrigin(sitk_image.GetOrigin())
    vtk_data = numpy_support.numpy_to_vtk(img_array.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_img.GetPointData().SetScalars(vtk_data)
    mesh_gen = vtk.vtkDiscreteMarchingCubes()
    mesh_gen.SetInputData(vtk_img)
    mesh_gen.SetValue(0, 1) # Look for mask value 1
    mesh_gen.Update()
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(mesh_gen.GetOutputPort())
    smoother.SetNumberOfIterations(iterations)
    smoother.BoundarySmoothingOn()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetPassBand(0.1) # 0.1 is standard for medical bone
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(output_path if output_path.endswith('.stl') else output_path + ".stl")
    writer.SetInputConnection(smoother.GetOutputPort())
    writer.Write()

# ---- Core functions ----
# Returns straightened vertebrae
def angle_correction(single_vertebrae_mask, adjacent_vertebrae_mask = None, Lumbar_mode = True):

    # Find bbox and calculate the center of the vertebrae in pixel and mm(physical) format
    single_vertebrae_mask_bbox = calc_bbox(single_vertebrae_mask)
    single_vertebrae_mask_center = (int((single_vertebrae_mask_bbox[0] + single_vertebrae_mask_bbox[3]) // 2), int((single_vertebrae_mask_bbox[1] + single_vertebrae_mask_bbox[4]) // 2), int((single_vertebrae_mask_bbox[2] + single_vertebrae_mask_bbox[5]) // 2))
    single_vertebrae_mask_center_physical = single_vertebrae_mask.TransformContinuousIndexToPhysicalPoint([single_vertebrae_mask_center[0], single_vertebrae_mask_center[1], single_vertebrae_mask_center[2]])

    # NOTE: PCA Does not work reliably
    # Working: The vertebrae is rotated -DEG deg to +DEG deg, and the 2D projection of the vertebrae is obtained. The 2D projection is obtained by performing bitwise OR operation for each slice along xy plane of the 3D volume. The area of largest hierarchy (vertebral foramen) is calculated in this 2D image, and the angle for which the hierarchy remains the largest is taken as sagittal angle
    max_area = 0
    max_angle = 0
    max_angle_index = 0
    max_hierarchy = None
    max_rotated_volume = None
    max_rotated_volume_bbox = None
    if Debug: max_image = None
    # Rotate the vertebrae from -DEG deg to +DEG deg
    sagittal_angle_range = [-30, 30] # NOTE: CONSTANT
    sagittal_angle_range[1], sagittal_mid_angle = 0, sagittal_angle_range[1]
    sagittal_current_angle_list = sagittal_angle_range.copy()
    sagittal_hierarchy_area = [0,0]
    first_iteration = 1
    # Rotate sagittal such that the vertebrae is rotated to the minimum to calculate the ideal angle
    while sagittal_mid_angle not in sagittal_current_angle_list:
        sagittal_current_angle_list[max_angle_index ^ 1] = sagittal_mid_angle
        max_prev_angle_index = max_angle_index
        for angle_index in range(len(sagittal_current_angle_list)):
            # After the first iteration, we only have to check the area of other angle, so maximum_previous_angle_index and first iteration are skipped
            if angle_index == max_prev_angle_index and not(first_iteration): continue
            if first_iteration: first_iteration = 0
            angle = sagittal_current_angle_list[angle_index]
            rotated_volume = rotate_volume(single_vertebrae_mask, [0, angle, 0], single_vertebrae_mask_center_physical)
            # Check if the maximum intensity in 0 (empty image)
            if sitk.MinimumMaximum(rotated_volume)[1] == 0: continue
            bbox = calc_bbox(rotated_volume)
            cv_image = get_2D_slice(rotated_volume, bbox[2], 2)
            # Obtain the 2D projection
            for z in range(bbox[2], bbox[2] + bbox[5] - 1):
                cv_image = np.bitwise_or(cv_image, get_2D_slice(rotated_volume, z, 2))
            # Find the hierarchy and produce only the largeest one
            contours, hierarchy = cv2.findContours(cv_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            hierarchies = []
            if hierarchy is not None:
                for i, h in enumerate(hierarchy[0]):
                    if h[3] != -1:
                        hierarchies.append(contours[i])
            if hierarchies:
                largest_heirarchy = max(hierarchies, key=cv2.contourArea)
                area = cv2.contourArea(largest_heirarchy)
                if area > max_area:
                    #max_spoinous_point = spoinous_point
                    max_area = area
                    max_angle = angle
                    max_angle_index = angle_index
                    max_hierarchy = largest_heirarchy
                    max_rotated_volume = sitk.Image(rotated_volume)
                    max_rotated_volume_bbox = bbox
                    if Debug: max_image = cv_image
            else:
                area = 0
        sagittal_mid_angle = (sagittal_current_angle_list[0] + sagittal_current_angle_list[1]) // 2

    # Sagittal angle operation (note that  max_rotated_volume has the sagitally straightened vertebrae)
    sagittal_angle = max_angle
    rotated_volume = max_rotated_volume
    adjacent_vertebrae_mask = rotate_volume(adjacent_vertebrae_mask, [0, sagittal_angle, 0], single_vertebrae_mask_center_physical)
    # Debug by drawing the bounary box and the image
    if Debug:
        cv2.imwrite("sagittal_2d_vertebrae.png", max_image)
        print("Debug: The calculated sagittal angle: ", sagittal_angle)

    # Find the foramen top and bottom point (useful for finding smallest pedicle slice)
    foramen_bottommost_point = tuple(max_hierarchy[max_hierarchy[:, :, 1].argmax()][0])

    # Finding coronal point by taking minAreaRect to the largest slice from foramen_bottommost_point
    rotated_volume_bbox = calc_bbox(rotated_volume)
    max_area = 0
    max_y = 0
    for y in range(foramen_bottommost_point[1], rotated_volume_bbox[1] + rotated_volume_bbox[4]):
        cv_image = get_2D_slice(rotated_volume, y, 1)
        contours, _ = cv2.findContours(cv_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > max_area:
                max_area = area
                max_y = y
                max_contour = largest_contour
                max_image = cv_image
    coronal_rect = cv2.minAreaRect(max_contour)
    coronal_angle = coronal_rect[2]
    if coronal_angle < -45: coronal_angle = -1 * (coronal_angle + 90)
    coronal_center = (round(coronal_rect[0][0]), max_y, round(coronal_rect[0][1]))
    coronal_center_physical = single_vertebrae_mask.TransformContinuousIndexToPhysicalPoint(coronal_center)
    if Debug:
        coronal_image = max_image.copy()
        cv2.circle(coronal_image, (coronal_center[0], coronal_center[2]), 4, 100, -1)
        box = cv2.boxPoints(coronal_rect)
        box = np.intp(box)
        cv2.polylines(coronal_image, [box], isClosed=True, color=150, thickness=2)
        cv2.imwrite("coronal_2d_vertebrae.png", coronal_image)
        print("Debug: The calculated coronal angle: ", coronal_angle)
    rotated_volume = rotate_volume(rotated_volume, [0, 0, coronal_angle], coronal_center_physical)
    adjacent_vertebrae_mask = rotate_volume(adjacent_vertebrae_mask, [0, 0, coronal_angle], coronal_center_physical)

    # Make coronal value accurate just like sagittal
    # Find bbox and calculate the center of the vertebrae in pixel and mm(physical) format
    rotated_volume_bbox = calc_bbox(rotated_volume)
    rotated_volume_center = (int((rotated_volume_bbox[0] + rotated_volume_bbox[3]) // 2), int((rotated_volume_bbox[1] + rotated_volume_bbox[4]) // 2), int((rotated_volume_bbox[2] + rotated_volume_bbox[5]) // 2))
    rotated_volume_center_physical = rotated_volume.TransformContinuousIndexToPhysicalPoint([rotated_volume_center[0], rotated_volume_center[1], rotated_volume_center[2]])

    # This foramen value is to define center for accurate coronal angle
    M = cv2.moments(max_hierarchy)
    if M["m00"] != 0:
        foramen_center_for_coronal = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        foramen_center_physical_for_coronal = single_vertebrae_mask.TransformContinuousIndexToPhysicalPoint([foramen_center_for_coronal[0], foramen_center_for_coronal[1], rotated_volume_center[2]])

    # Working: The vertebrae is rotated -DEG deg to +DEG deg, and the 2D projection of the vertebrae is obtained. The 2D projection is obtained by performing bitwise OR operation for each slice along xy plane of the 3D volume. The area of largest hierarchy (vertebral foramen) is calculated in this 2D image, and the angle for which the hierarchy remains the largest is taken as sagittal angle
    max_area = 0
    max_angle = 0
    max_angle_index = 0
    max_hierarchy = None
    max_rotated_volume = None
    max_rotated_volume_bbox = None
    if Debug: max_image = None
    # Rotate the vertebrae from -DEG deg to +DEG deg
    rotated_volume_init = sitk.Image(rotated_volume)
    sagittal_angle_range = [-30, 30] # NOTE: CONSTANT
    sagittal_angle_range[1], sagittal_mid_angle = 0, sagittal_angle_range[1]
    sagittal_current_angle_list = sagittal_angle_range.copy()
    sagittal_hierarchy_area = [0,0]
    first_iteration = 1
    # Rotate sagittal such that the vertebrae is rotated to the minimum to calculate the ideal angle
    while sagittal_mid_angle not in sagittal_current_angle_list:
        sagittal_current_angle_list[max_angle_index ^ 1] = sagittal_mid_angle
        max_prev_angle_index = max_angle_index
        for angle_index in range(len(sagittal_current_angle_list)):
            # After the first iteration, we only have to check the area of other angle, so maximum_previous_angle_index and first iteration are skipped
            if angle_index == max_prev_angle_index and not(first_iteration): continue
            if first_iteration: first_iteration = 0
            angle = sagittal_current_angle_list[angle_index]
            rotated_volume = rotate_volume(rotated_volume_init, [0, 0, angle], coronal_center_physical)
            # Check if the maximum intensity in 0 (empty image)
            if sitk.MinimumMaximum(rotated_volume)[1] == 0: continue
            bbox = calc_bbox(rotated_volume)
            cv_image = get_2D_slice(rotated_volume, bbox[2], 2)
            # Obtain the 2D projection
            for z in range(bbox[2], bbox[2] + bbox[5] - 1):
                cv_image = np.bitwise_or(cv_image, get_2D_slice(rotated_volume, z, 2))
            # Find the hierarchy and produce only the largeest one
            contours, hierarchy = cv2.findContours(cv_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            hierarchies = []
            if hierarchy is not None:
                for i, h in enumerate(hierarchy[0]):
                    if h[3] != -1:
                        hierarchies.append(contours[i])
            if hierarchies:
                largest_heirarchy = max(hierarchies, key=cv2.contourArea)
                area = cv2.contourArea(largest_heirarchy)
                if area > max_area:
                    #max_spoinous_point = spoinous_point
                    max_image = cv_image
                    max_area = area
                    max_angle = angle
                    max_angle_index = angle_index
                    max_hierarchy = largest_heirarchy
                    max_rotated_volume = sitk.Image(rotated_volume)
                    max_rotated_volume_bbox = bbox
                    if Debug: max_image = cv_image
            else:
                area = 0
        sagittal_mid_angle = (sagittal_current_angle_list[0] + sagittal_current_angle_list[1]) // 2

    # Sagittal angle operation (note that  max_rotated_volume has the sagitally straightened vertebrae)
    accurate_coronal_angle = max_angle
    rotated_volume = max_rotated_volume
    adjacent_vertebrae_mask = rotate_volume(adjacent_vertebrae_mask, [0, 0, accurate_coronal_angle], coronal_center_physical)
    if Debug:
        cv2.imwrite("accurate_coronal_2d_vertebrae.png", max_image)
        print("Debug: The calculated accurate coronal angle: ", accurate_coronal_angle)

    # Working: Axial angle is simply obtained by calculating angle of two points, the tip of the vertebrae (tip of sponious process) and center of the found largest hierarchy (the center of vertebral foramen)
    # Calculating axial angle
    # If approximation should happen especially for thorasic, axially
    if not(Lumbar_mode):
        M = cv2.moments(max_hierarchy)
        if M["m00"] != 0:
            foramen_center = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            foramen_center_physical = single_vertebrae_mask.TransformContinuousIndexToPhysicalPoint([foramen_center[0], foramen_center[1], single_vertebrae_mask_center[2]])
        foramen_bottommost_point = tuple(max_hierarchy[max_hierarchy[:, :, 1].argmax()][0])
        single_vertebrae_body_mask = max_image.copy()
        single_vertebrae_body_mask[:foramen_bottommost_point[1], :] = 0
        M = cv2.moments(single_vertebrae_body_mask)
        if M["m00"] != 0:
            vertebrae_body_mask_centroid = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            vertebrae_body_mask_centroid_physical = single_vertebrae_mask.TransformContinuousIndexToPhysicalPoint([foramen_center[0], foramen_center[1], single_vertebrae_mask_center[2]])
        approx_axial_angle = math.degrees(math.atan2(abs(foramen_center[0] - vertebrae_body_mask_centroid[0]), abs(foramen_center[1] - vertebrae_body_mask_centroid[1])))
        if foramen_center[0] > vertebrae_body_mask_centroid[0]: approx_axial_angle *= -1
        rotated_volume = rotate_volume(rotated_volume, [approx_axial_angle, 0, 0], vertebrae_body_mask_centroid_physical)
        adjacent_vertebrae_mask = rotate_volume(adjacent_vertebrae_mask, [approx_axial_angle, 0, 0], vertebrae_body_mask_centroid_physical)
        # Calculate bbox for new rotated vertebrae and repeat the same thing happened in calculating sagittal angle
        bbox = calc_bbox(rotated_volume)
        cv_image = get_2D_slice(rotated_volume, bbox[2], 2)
        # Obtain the 2D projection
        for z in range(bbox[2], bbox[2] + bbox[5] - 1):
            cv_image = np.bitwise_or(cv_image, get_2D_slice(rotated_volume, z, 2))
        # Find the hierarchy and produce only the largeest one
        contours, hierarchy = cv2.findContours(cv_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hierarchies = []
        if hierarchy is not None:
            for i, h in enumerate(hierarchy[0]):
                if h[3] != -1:
                    hierarchies.append(contours[i])
        if hierarchies:
            largest_heirarchy = max(hierarchies, key=cv2.contourArea)
            area = cv2.contourArea(largest_heirarchy)
            max_hierarchy = largest_heirarchy
            max_rotated_volume = sitk.Image(rotated_volume)
            max_rotated_volume_bbox = bbox
            if Debug: max_image = cv_image
        if Debug:
            debug_image = max_image.copy()
            cv2.line(debug_image, foramen_center, vertebrae_body_mask_centroid,150, 2)
            cv2.circle(debug_image, foramen_center, 4, 150, -1)
            cv2.circle(debug_image, vertebrae_body_mask_centroid, 3, 150, -1)
            cv2.imwrite("approximate_axial_2d_vertebrae.png", debug_image)
            print("Debug: The calculated approximate axial angle: ", approx_axial_angle)

    # Obtan the spoinous process point
    cv_image = get_2D_slice(rotated_volume, max_rotated_volume_bbox[1], 1)
    contours, _ = cv2.findContours(cv_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        spoinous_topmost_point = int(rect[0][0]), max_rotated_volume_bbox[1]
    # Obtain the center of the vertebral foramen
    # Calculate coordinates (add a check to avoid division by zero for tiny noise artifacts)
    M = cv2.moments(max_hierarchy)
    if M["m00"] != 0:
        foramen_center = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        foramen_center_physical = single_vertebrae_mask.TransformContinuousIndexToPhysicalPoint([foramen_center[0], foramen_center[1], single_vertebrae_mask_center[2]])
        axial_angle = math.degrees(math.atan2(abs(spoinous_topmost_point[0] - foramen_center[0]), abs(spoinous_topmost_point[1] - foramen_center[1])))
    if spoinous_topmost_point[0] > foramen_center[0]: axial_angle *= -1
    rotated_volume = rotate_volume(rotated_volume, [axial_angle, 0, 0], foramen_center_physical)
    adjacent_vertebrae_mask = rotate_volume(adjacent_vertebrae_mask, [axial_angle, 0, 0], foramen_center_physical)
    if Debug:
        debug_image = max_image.copy()
        cv2.line(debug_image, spoinous_topmost_point, foramen_center, 150, 2)
        cv2.circle(debug_image, spoinous_topmost_point, 4, 150, -1)
        cv2.circle(debug_image, foramen_center, 3, 150, -1)
        cv2.imwrite("axial_2d_vertebrae.png", debug_image)
        print("Debug: The calculated axial angle: ", axial_angle)

    # Calculate foramen points
    M = cv2.moments(max_hierarchy)
    if M["m00"] != 0:
        foramen_center = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    foramen_topmost_point = tuple(max_hierarchy[max_hierarchy[:, :, 1].argmin()][0])
    foramen_bottommost_point = tuple(max_hierarchy[max_hierarchy[:, :, 1].argmax()][0])
    if Debug:
        axial_image = max_image.copy()
        cv2.circle(axial_image, foramen_topmost_point[:2], 4, 100, -1)
        cv2.circle(axial_image, foramen_bottommost_point[:2], 4, 100, -1)
        cv2.imwrite("foramen_points.png", debug_image)
 
    # Rotate adjacent vertebrae sagitally and axially (in order)
    if adjacent_vertebrae_mask != None: pass

    # Return the rotated vertebrae along with the foramen_topmost_point for accurate pedicle segmentation
    return rotated_volume, adjacent_vertebrae_mask, [foramen_topmost_point, foramen_center, foramen_bottommost_point]

# Get smallest slice and filter them as left or right passthrough pedicle mask
def passthrough_slice(single_vertebrae_mask, foramen_points):
    min_area = float('inf')
    min_y = foramen_points[1][1]
    for y in range(foramen_points[0][1], foramen_points[2][1]):
        cv_image = get_2D_slice(single_vertebrae_mask, y, 1)
        contours, _ = cv2.findContours(cv_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            area = sum(cv2.contourArea(contour) for contour in contours)
            if area < min_area:
                min_area = area
                min_y = y
                min_image = cv_image
    passthrough_slice_list = [min_image.copy(), min_image.copy()]
    passthrough_slice_list[0][:, foramen_points[1][0]:] = 0
    passthrough_slice_list[1][:, :foramen_points[1][0]] = 0
    if Debug:
        print("Debug: The minimum slice is obtained in: ", min_y)
        cv2.imwrite("passthrough_slice_1.png", passthrough_slice_list[0])
        cv2.imwrite("passthrough_slice_2.png", passthrough_slice_list[1])
    return passthrough_slice_list, min_y

def height_map(single_vertebrae_mask, slice_y, dilation_iterations = 1, projection_largest_contour = True, adjacent_vertebrae_mask = None, Lumbar_mode = False):
    # Cut off the vertebrae_body
    height_vertebrae_mask = sitk.Image(single_vertebrae_mask)
    height_vertebrae_mask[:, slice_y:, :] = 0
    bbox = calc_bbox(height_vertebrae_mask)
    # Create a height map with the xz dimensions
    image_size = height_vertebrae_mask.GetSize()
    height_map_array = np.zeros((image_size[0], image_size[2]))
    gradient_map_array = np.zeros((image_size[0], image_size[2]))
    # Make a initial slice for top-down 2D projection
    single_vertebrae_2d_projection = get_2D_slice(height_vertebrae_mask, bbox[0], 1)
    if adjacent_vertebrae_mask != None: adjacent_vertebrae_2d_projection = get_2D_slice(adjacent_vertebrae_mask, bbox[0], 1)
    lookback_value = 8 # NOTE: CONSTANT
    #gradient_map_array = np.zeros((bbox[3] + 1, bbox[5] + 1))
    for y in range(bbox[1] + bbox[4] - 1, bbox[1], -1):
        cv_image = get_2D_slice(height_vertebrae_mask, y, 1)
        single_vertebrae_2d_projection = np.bitwise_or(single_vertebrae_2d_projection, cv_image)
        if adjacent_vertebrae_mask != None:
            adjacent_cv_image = get_2D_slice(adjacent_vertebrae_mask, y, 1)
            adjacent_vertebrae_2d_projection = np.bitwise_or(adjacent_vertebrae_2d_projection, adjacent_cv_image)
        coordinates = cv2.findNonZero(cv_image)
        # Fill height map with actual height values
        for coord in coordinates:
            coord = np.atleast_2d(coord)
            height_map_array[coord[0][0]][coord[0][1]] = y
        for coord in coordinates:
            coord = np.atleast_2d(coord)
            max_height = 0
            if height_map_array[coord[0][0]][coord[0][1] - lookback_value] > max_height:
                max_height = height_map_array[coord[0][0]][coord[0][1] - lookback_value]
            if max_height: 
                gradient_map_array[coord[0][0]][coord[0][1]] = max(0, max_height - y + 1)
            else: gradient_map_array[coord[0][0]][coord[0][1]] = 1
    # Fill tiles with the maximum difference in height it can find around it with pure white
    if Lumbar_mode:
        map_threshold = 1 # NOTE: CONSTANT
    else:
        map_threshold = 0 # Takes anything greater than 0 (takes full mask from gradient mask)
    _, sap_map_array = cv2.threshold(gradient_map_array.astype(np.uint8), map_threshold, 255, cv2.THRESH_BINARY)
    sap_map_array = np.transpose(sap_map_array)
    height_map_array = np.transpose(height_map_array)
    # Dilate SAP
    #ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    #sap_map_array = cv2.dilate(sap_map_array, ellipse_kernel, iterations = dilation_iterations) # NOTE: CONSTANT
    # Delete the SAP portion by taking and and xor
    if Lumbar_mode:
        common_array = cv2.bitwise_and(single_vertebrae_2d_projection, sap_map_array.astype(np.uint8))
        single_vertebrae_2d_projection = cv2.bitwise_xor(single_vertebrae_2d_projection, common_array)
    common_array = cv2.bitwise_and(single_vertebrae_2d_projection, adjacent_vertebrae_2d_projection)
    single_vertebrae_2d_projection = cv2.bitwise_xor(single_vertebrae_2d_projection, common_array)

    # Take the largest contour # NOTE: Test this, since we take largest contour, it may miss vital points
    if projection_largest_contour:
        contours, _ = cv2.findContours(single_vertebrae_2d_projection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        single_vertebrae_2d_projection= np.zeros_like(single_vertebrae_2d_projection)
        cv2.drawContours(single_vertebrae_2d_projection, [largest_contour], -1, 255, thickness=cv2.FILLED)
    if Debug:
        gradient_map_array = np.transpose(gradient_map_array)
        cv2.imwrite('height_map_array.png', height_map_array.astype(np.uint8))
        cv2.imwrite('gradient_map_array.png', gradient_map_array.astype(np.uint8) * 10)
        cv2.imwrite('sap_map_array.png',sap_map_array.astype(np.uint8) * 10)
        cv2.imwrite('single_vertebrae_2d_projection.png',single_vertebrae_2d_projection.astype(np.uint8) * 10)
        cv2.imwrite('adjacent_vertebrae_2d_projection.png',adjacent_vertebrae_2d_projection.astype(np.uint8) * 10)

    return single_vertebrae_2d_projection.astype(np.uint8), height_map_array

# Find the best screw trajectory and also return a jig line along with it
def best_screw_trajectory(single_vertebrae_mask, top_slice, height_map_array, passthrough_slice, passthrough_slice_y, foramen_points, length):

    # Prepare cv_images
    bbox = calc_bbox(single_vertebrae_mask)
    cv_images_dict = {}
    for y in range(bbox[1], bbox[1] + bbox[4]):
        cv_image = get_2D_slice(single_vertebrae_mask, y)
        dist_map = cv2.distanceTransform(cv_image, cv2.DIST_L2, cv2.DIST_MASK_5)
        cv_images_dict.update({y: dist_map})

    # Prepare coordinates
    top_coordinates = cv2.findNonZero(top_slice)
    bottom_coordinates = cv2.findNonZero(passthrough_slice)
    
    # Prepare variables
    p_min_distance_list = []
    p_min_points = []
    p_min_coord = []

    # Generate list of points as numpy list
    single_vertebrae_mask_np = sitk.GetArrayFromImage(single_vertebrae_mask)
    # Generate lines
    for t_coord in top_coordinates:
        # Find the highest y level of the top coordinate to generate 3D point
        t_coord = np.atleast_2d(t_coord)
        t_coord_3d = [int(t_coord[0][0]), height_map_array[int(t_coord[0][1]), int(t_coord[0][0])], int(t_coord[0][1])]
        for b_coord in bottom_coordinates:
          # Generate 3D point for bottom coordinate
            b_coord = np.atleast_2d(b_coord)
            b_coord_3d = [int(b_coord[0][0]), passthrough_slice_y, int(b_coord[0][1])]
            # Generate line points
            line_points = generate_3d_line_by_length(t_coord_3d, b_coord_3d, length)
            # Check if the line lies within the single_vertebrae_mask
            line_points_int = np.round(line_points).astype(int)
            x_idx = line_points_int[:, 0]
            y_idx = line_points_int[:, 1]
            z_idx = line_points_int[:, 2]
            in_bounds = (
                (z_idx >= 0) & (z_idx < single_vertebrae_mask_np.shape[0]) &
                (y_idx >= 0) & (y_idx < single_vertebrae_mask_np.shape[1]) &
                (x_idx >= 0) & (x_idx < single_vertebrae_mask_np.shape[2])
            )
            if not np.all(in_bounds): line_lies_inside = False
            else:
                voxel_values = single_vertebrae_mask_np[z_idx, y_idx, x_idx]
                line_lies_inside = np.all(voxel_values == 1)
            probable_path = True
            if not(line_lies_inside):
                probable_path = False
                continue
            if not(np.any(line_points[:, 1] >= foramen_points[2][1])):
                probable_path = False
                continue
            if not(np.any(line_points[:, 1] < passthrough_slice_y)):
                probable_path = False
                continue

            # Start checking
            # NOTE: please note that probable path may be skipped because of previous code. During debugging disable the previous code.
            t_min_distance_list = []
            for point in line_points:
                y_point = round(point[1])
                distance = cv_images_dict[y_point][round(point[2]), round(point[0])]
                if False: #distance > 0:
                    probable_path = False
                    break
                else:
                    distance = abs(distance)
                    #if y_point >= foramen_points[0][1]: #and y_point <= foramen_points[2][1]:
                    t_min_distance_list.append(distance)

            # If all points exist within the vertebrae, then take the line with the largest minimum distance (best worst-case scenario)
            if probable_path:
                t_min_distance_list.sort()
                if p_min_distance_list == []: 
                    p_min_distance_list = t_min_distance_list.copy()
                    p_min_points = line_points.copy()
                if t_min_distance_list > p_min_distance_list:
                    p_min_distance_list = t_min_distance_list.copy()
                    p_min_points = line_points.copy()
                    p_min_coord = [t_coord_3d, b_coord_3d]
                elif t_min_distance_list < p_min_distance_list:
                    probable_path = False
    # Jig_line_points for generating jig
    jig_line_points = generate_3d_line_by_length(p_min_points[-1], p_min_points[0], 256) # NOTE: CONSTANT
    return p_min_points, jig_line_points

# Created a 3D printable jig
def create_jig(single_vertebrae_mask, adjacent_vertebrae_mask, p_min_points, jig_line_points, passthrough_slice_y):

    # Vertebrae mask combined
    single_vertebrae_mask = single_vertebrae_mask | adjacent_vertebrae_mask
    # Working: Create a 3D -> 2D -> 3D projection by slicing and coverting each 3D slice to 2D slice, appending them in a list, and convertng the total list back to 3D, to get a shadow-cast volume.
    bbox = calc_bbox(single_vertebrae_mask)
    single_vertebrae_2d_projection = get_2D_slice(single_vertebrae_mask, bbox[1], 1)
    shadow_projection_list = []
    image_size = single_vertebrae_mask.GetSize()
    empty_image_array = get_2D_slice(single_vertebrae_mask, bbox[1] - 1)
    empty_image_array = np.where(empty_image_array > 0, 1, 0).astype(np.uint8)
    # Create bbox for adjacent vertebrae mask
    adjacent_bbox = calc_bbox(adjacent_vertebrae_mask)
    adjacent_mid_z = adjacent_bbox[2] + adjacent_bbox[5] // 2
    # Create empty slice and append for slices less than object's start height and greater than object'ss end height
    for i in range(bbox[1]):
        shadow_projection_list.append(empty_image_array)
    for y in range(bbox[1], passthrough_slice_y):
        cv_image = get_2D_slice(single_vertebrae_mask, y, 1)
        cv_image = np.where(cv_image > 0, 1, 0).astype(np.uint8)
        single_vertebrae_2d_projection = cv2.bitwise_or(single_vertebrae_2d_projection, cv_image)
        shadow_projection_list.append(single_vertebrae_2d_projection)
    for i in range(passthrough_slice_y, image_size[1]):
        shadow_projection_list.append(empty_image_array)
    shadow_volume_np = np.stack(shadow_projection_list, axis=1)
    shadow_volume = sitk.GetImageFromArray(shadow_volume_np)
    shadow_volume.CopyInformation(single_vertebrae_mask)
    # Define jig thickness
    jig_thickness = 5 # NOTE: CONSTANT
    # Create a shifted array, by appending more empty slice in the last, and pushing object slice above
    for i in range(jig_thickness - 1):
        del shadow_projection_list[0]
        shadow_projection_list.append(empty_image_array)
    shifted_shadow_volume_np = np.stack(shadow_projection_list, axis=1)
    shifted_shadow_volume = sitk.GetImageFromArray(shifted_shadow_volume_np)
    shifted_shadow_volume.CopyInformation(single_vertebrae_mask)

    connector_radius = 3 # NOTE: CONSTANT
    constant_height = 10 # NOTE: CONSTANT
    connector_constant_height = 4 # NOTE: CONSTANT
    jig_outer_list = []
    jig_inner_list = []
    jig_connector_points = []
    jig_list = []

    # Define the points required, generate required volumes
    for jig_line_index in range(len(jig_line_points)):
        jig_line = jig_line_points[jig_line_index]
        jig_inner = generate_trajectory_volume(single_vertebrae_mask, jig_line.copy(), inner_radius)
        jig_line = jig_line[jig_line[:, 1] > (bbox[1] - constant_height)]
        jig_outer = generate_trajectory_volume(single_vertebrae_mask, jig_line.copy(), outer_radius)
        jig_line = jig_line[jig_line[:, 1] > (bbox[1] - connector_constant_height)]
        jig_outer_list.append(jig_outer)
        jig_inner_list.append(jig_inner)
        jig_connector_points.append(jig_line[-1])
        jig_line = jig_line[jig_line[:, 1] < (p_min_points[jig_line_index][0][1] - jig_thickness)]
        jig_list.append(jig_line)

    start = np.array(jig_connector_points[0], dtype=float)
    stop = np.array(jig_connector_points[1], dtype=float)
    connector_points = generate_3d_line_by_length(start, stop, int(np.linalg.norm(start - stop)))
    connector = generate_trajectory_volume(single_vertebrae_mask, connector_points, connector_radius)

    shifted_shadow_volume[:int(jig_list[0][0][0]) - outer_radius, :, :] = 0
    if (int(jig_list[0][0][0]) + outer_radius) < (int(jig_list[1][0][0]) - outer_radius):
        shifted_shadow_volume[int(jig_list[0][0][0]) + outer_radius: int(jig_list[1][0][0]) - outer_radius, :, :] = 0
    shifted_shadow_volume[int(jig_list[1][0][0]) + outer_radius:, :, :] = 0
    shifted_shadow_volume[:, :, adjacent_mid_z:] = 0
    
    # Perform basic operation to get the jig (shifted_shadow_volume)
    shifted_shadow_volume = shifted_shadow_volume | jig_outer_list[0] | jig_outer_list[1] | connector
    shifted_shadow_volume = shifted_shadow_volume ^ (shifted_shadow_volume & (jig_inner_list[0] | jig_inner_list[1]))
    shifted_shadow_volume = (shifted_shadow_volume ^ (shifted_shadow_volume & shadow_volume))
    shifted_shadow_volume[:, passthrough_slice_y - jig_thickness:, :] = 0
    shifted_shadow_volume = largest_object(shifted_shadow_volume)
    trajectory_0 = generate_trajectory_volume(single_vertebrae_mask, p_min_points[0].copy(), s_radius)
    trajectory_1 = generate_trajectory_volume(single_vertebrae_mask, p_min_points[1].copy(), s_radius)
    if Debug:
        sitk.WriteImage(shifted_shadow_volume, "jig_base_layer_volume.nii")
        sitk.WriteImage(trajectory_0, "trajectory_0.nii")
        sitk.WriteImage(trajectory_1, "trajectory_1.nii")
    return shifted_shadow_volume

if __name__ == "__main__":
    # Ensure the required folders are present
    data_dir = Path("data")
    data_required_folders = ("downloads", "generated", "nii_files", "segmentations", "outputs")
    for folder_name in data_required_folders:
        subfolder = data_dir / folder_name
        subfolder.mkdir(parents=True, exist_ok=True)

    # Read Arguments
    input_file = str(Path(sys.argv[1]))
    selected_vertebraes = sys.argv[2].split('+')
    s_height = math.ceil(float(sys.argv[3]) / Slice_MM)
    s_radius = math.ceil((float(sys.argv[4]) / 2) / Slice_MM)
    outer_radius = math.ceil(int((float(sys.argv[6]) / 2) / Slice_MM))
    inner_radius = math.ceil(int((float(sys.argv[5]) / 2) / Slice_MM))
    if Debug: print(f"Note: Height taken as {s_height}\nNote: Radius taken as {s_radius}\nNote: Jig Outer Radius taken as {outer_radius}\nNote: Jig Inner Radius taken as {inner_radius}")

    # Convert DICOM to .nii if possible
    if Debug: print('File Provided: ', input_file)
    if os.path.isdir(input_file):
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(input_file)
        if series_ids:
            largest_series_id = max(series_ids, key=lambda sid: len(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(input_file, sid)))
            dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(input_file, largest_series_id)
            img = sitk.ReadImage(dicom_names)
    else:
        img = sitk.ReadImage(input_file)


    # Resampling
    if Run_Resampler:
        img = resample_ct_volume(img, Slice_MM)
        if Debug: print("Resampled Image:", input_file)
    sitk.WriteImage(img, str(Path("data/outputs/resampled_ct_scan.nii.gz")), useCompression=True)

    # Totalsegmentator
    if Run_Totalsegmentator:
        totalsegmentator(
            str(Path("data/outputs/resampled_ct_scan.nii.gz")), 
            str(Path("data/segmentations/")), 
            task="vertebrae_mr", 
        )
        if Debug: print('TotalSegmentator mr complete: ', input_file)

    # Initialize variable to reconstruct spine
    spine_mask = sitk.ReadImage(str(Path(f"data/segmentations/{Spine_Vertebraes[0]}.nii.gz")))
    # List each vertebrae and update Single_Vertebrae_Dict [Used to make negative which required predecessive and successive vertebrae]
    for vertebrae_name in Spine_Vertebraes:
        if (vertebrae_name + ".nii.gz") in os.listdir(str(Path("data/segmentations/"))):
            if Debug: print('Adding single_vertebrae_mask: ', vertebrae_name)
            single_vertebrae_mask = sitk.ReadImage(str(Path(f"data/segmentations/{vertebrae_name}.nii.gz")))
            spine_mask = spine_mask | single_vertebrae_mask
            Single_Vertebrae_Dict.update({vertebrae_name: single_vertebrae_mask})
    if Debug: sitk.WriteImage(spine_mask, str(Path("data/generated/spine_output.nii.gz")))
    if STL_Conversion: sitk_to_stl(spine_mask, str(Path("data/generated/spine.stl")))

    # Change directory and loop
    vertebrae_screw_gen_truth = ""
    directory_listing = os.listdir(str(Path("data/generated")))
    for vertebrae_name in selected_vertebraes:
        if vertebrae_name in Single_Vertebrae_Dict.keys():
            if vertebrae_name not in directory_listing:
                os.mkdir(str(Path(f"data/generated/{vertebrae_name}")))
            os.chdir(str(Path(f"data/generated/{vertebrae_name}")))
            if Debug: print(f"Starting screw placement for: {vertebrae_name}")
            single_vertebrae_mask = largest_object(Single_Vertebrae_Dict[vertebrae_name])
            spine_vertebrae_index = Spine_Vertebraes.index(vertebrae_name) - 1, Spine_Vertebraes.index(vertebrae_name) + 1
            adjacent_vertebrae_mask = largest_object(Single_Vertebrae_Dict[Spine_Vertebraes[spine_vertebrae_index[0]]])
            # Check if the vertebrae is in lumbar / thorasic part
            if vertebrae_name in ['vertebrae_T12','vertebrae_L1', 'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5']: Lumbar_mode = True
            else: Lumbar_mode = False
            if Debug: print("Lumbar mode: ", Lumbar_mode)
            rotated_vertebrae, rotated_adjacent_vertebrae, foramen_points = angle_correction(single_vertebrae_mask, adjacent_vertebrae_mask, Lumbar_mode)
            if Debug: sitk.WriteImage(rotated_vertebrae, "rotated_vertebrae.nii")
            if Debug: sitk.WriteImage(rotated_adjacent_vertebrae, "rotated_adjacent_vertebrae.nii")
            passthrough_slice_list, passthrough_slice_y = passthrough_slice(rotated_vertebrae, foramen_points)
            top_slice, height_map_array = height_map(rotated_vertebrae, passthrough_slice_y, 1, True, rotated_adjacent_vertebrae, Lumbar_mode)
            if Debug: print("Trajectory 0 generating")
            trajectory_points_0, jig_line_points_0 = best_screw_trajectory(rotated_vertebrae, top_slice, height_map_array, passthrough_slice_list[0], passthrough_slice_y, foramen_points, s_height)
            if Debug: print("Trajectory 1 generating")
            trajectory_points_1, jig_line_points_1 = best_screw_trajectory(rotated_vertebrae, top_slice, height_map_array, passthrough_slice_list[1], passthrough_slice_y, foramen_points, s_height)
            if len(trajectory_points_0) != 0 and len(trajectory_points_1) != 0:
                if Debug: print("Jig generating")
                jig_volume = create_jig(rotated_vertebrae, rotated_adjacent_vertebrae, (trajectory_points_0, trajectory_points_1), (jig_line_points_0, jig_line_points_1), passthrough_slice_y)
                vertebrae_screw_gen_truth += "1"
                if STL_Conversion: sitk_to_stl(rotated_vertebrae, "rotated_vertebrae.stl")
                if STL_Conversion: sitk_to_stl(jig_volume, "jig_base_layer_volume.stl")
            else:
                if Debug: print("Jig generation FAILED")
                vertebrae_screw_gen_truth += "0"
            os.chdir(str(Path("../../../")))
    for i in range(len(selected_vertebraes)):
        if vertebrae_screw_gen_truth[i] == "1": status_string = "Success"
        else: status_string = "Failed"
        print(f"Status of {selected_vertebraes[i]}: {status_string}")

    # Make archive
    shutil.make_archive(
        base_name = str(Path("data/downloads/generated_vertebraes")), 
        format = "zip",
        root_dir = str(Path("data/generated"))
    )
    # Delete folders
    if not(Testing):
        gen_dir = Path("data/generated")
        shutil.rmtree(str(Path("data/generated")))
        gen_dir.mkdir(parents=True, exist_ok=True)
    with open(str(Path("data/vertebrae_screw_gen_truth.txt")), "w") as file:
        file.write(vertebrae_screw_gen_truth)
    print("Program Completed :)")
