---

title: "3D Lung Visualization"
date: 2024-6-5
id: 3
author: "Afnan K Salal"
authorGithub: "https://github.com/afnanksalal"
tags:
  - Python
  - Streamlit
  - DICOM
  - 3D Visualization
  - Lung Segmentation
  - Medical Imaging
  - Data Analysis
  - Interactive Web Apps
  - U-Net
  - Deep Learning
  - AI
---

##  Peering into the Lungs: A Comprehensive Guide to 3D Lung Visualization with Streamlit, Lungmask, and AI

Welcome to a world where data comes to life! This blog post will be your comprehensive guide to building a powerful 3D lung visualization application using Python, Streamlit, and the magic of AI. 

We'll journey through the fascinating world of medical imaging, exploring the intricacies of DICOM files, the power of the `lungmask` library, and the elegance of 3D visualization with Plotly. 

This is a deep dive for beginners, covering every step, every line of code, and all the essential concepts to empower you to create your own medical imaging applications. 

**Why is 3D Lung Visualization So Important?**

Visualizing and analyzing 3D lung data is revolutionizing medical imaging, providing unparalleled insights that enhance diagnosis, treatment, and research:

*   **Diagnosis:** Imagine a doctor looking at a 2D image of your lungs. It's like trying to understand a complex puzzle by looking at only one piece. 3D visualization allows them to see the entire lung structure, making it much easier to detect subtle abnormalities, tumors, nodules, or areas of infection. 
*   **Treatment Planning:**  Imagine a surgeon planning a delicate surgery on your lungs. 3D visualization provides a detailed, 360-degree view, helping them precisely identify the areas to operate on and minimize the risks.
*   **Research:**  Imagine scientists studying lung diseases.  3D visualization helps them analyze lung structures and their changes over time, leading to groundbreaking discoveries about lung function and treatments.

**Ready to Explore the Lungs?** Let's begin!

### 1. Understanding the Basics: DICOM, 3D Visualization, and Segmentation

**1.1  DICOM: The Language of Medical Imaging**

DICOM stands for **Digital Imaging and Communications in Medicine.** It's a standardized format for storing, transmitting, and displaying medical images. Think of it as the universal language of medical imaging, allowing doctors, hospitals, and researchers to share and analyze data seamlessly. 

*   **Why is DICOM important?**
    *   **Consistency:** DICOM ensures that all medical images are formatted in a consistent way, making them easily readable and interpretable by different systems and software.
    *   **Interoperability:**  DICOM allows medical devices and software from different manufacturers to communicate and exchange data smoothly.
    *   **Data Integrity:**  DICOM includes mechanisms for verifying the authenticity and integrity of medical images, preventing data corruption and ensuring the accuracy of diagnoses.

**1.2  3D Visualization: Bringing Data to Life**

Imagine trying to understand a complex object like a sculpture by looking at a single photograph. It's challenging! 3D visualization allows us to view objects from multiple angles, revealing their full structure and depth.

*   **Why is 3D visualization essential?**
    *   **Enhanced Understanding:** 3D visualization helps us see and interpret complex data in a more intuitive and engaging way.
    *   **Improved Insights:**  By examining data from different perspectives, we can gain a deeper understanding of its patterns, relationships, and anomalies.
    *   **Effective Communication:** 3D visualizations can effectively communicate complex information to a broader audience, including medical professionals, researchers, and even patients.

**1.3 Lung Segmentation: Isolating the Lungs**

Lung segmentation is the process of separating the lung regions from the rest of the body in a medical image. Think of it like carefully cutting out a lung shape from a picture.  Accurate lung segmentation is crucial for 3D visualization because it allows us to focus on the lungs and analyze them in detail.

*   **Why is lung segmentation so important?**
    *   **Focused Analysis:**  By segmenting the lungs, we can eliminate distractions and focus our analysis on the specific areas of interest.
    *   **Enhanced Accuracy:**  Accurate lung segmentation improves the precision of measurements, such as lung volume or the size of tumors.
    *   **Automated Analysis:**  Segmentation enables us to automate many tasks, such as volume calculations and abnormality detection, saving time and effort for medical professionals.

### 2. Project Setup: Building Your Foundation

**2.1 Install Essential Libraries**

First, we need the right tools for our job. Open your terminal and install the necessary libraries using `pip`:

```bash
pip install streamlit numpy scikit-image plotly lungmask SimpleITK
```

*   **Streamlit:**  A powerful Python library for building interactive web applications quickly. We'll use it to create a user interface for uploading DICOM files and displaying the 3D lung visualization.
*   **NumPy:** The fundamental library for numerical computation in Python. We'll rely on NumPy arrays to represent and manipulate the lung data.
*   **Scikit-image:** A collection of image processing algorithms in Python. We'll use its `marching_cubes` function to extract the surface mesh of the segmented lung volume.
*   **Plotly:** A library for creating interactive, web-based visualizations. We'll use Plotly to generate our 3D lung visualization.
*   **Lungmask:** A specialized library designed for automatic lung segmentation in DICOM images. `Lungmask` will be our key tool for accurately isolating the lungs from the surrounding tissues.
*   **SimpleITK:** A powerful toolkit for image analysis, especially for handling medical images like DICOM files. We'll use `SimpleITK` to read and manipulate the DICOM data.

**2.2 Create a `main.py` File**

Create a new Python file named `main.py` in your project directory. This file will contain all the code for your Streamlit application.

### 3. The Code Explained: Step-by-Step Breakdown

```python
import streamlit as st
import numpy as np
from skimage import measure
import plotly.graph_objects as go
from lungmask import LMInferer
import SimpleITK as sitk
import tempfile
import os

# --- Load Lung Data from Uploaded DICOM Files ---
def load_dicom_files(uploaded_files):
    """Loads a list of DICOM files and stacks the lungs after segmentation."""
    inferer = LMInferer()
    modified_image_masks = []
    for uploaded_file in uploaded_files:
        try:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_filename = temp_file.name
            
            # Read the DICOM data from the temporary file
            input_image = sitk.ReadImage(temp_filename)
            segmentation = inferer.apply(input_image)  # Apply lungmask

            # Create a mask where lungs are True, background is False
            lung_mask = segmentation != 0

            # Apply mask to the original image array
            image_array = sitk.GetArrayFromImage(input_image)
            image_array[~lung_mask] = 0  # Set background pixels to black

            modified_image_masks.append(image_array)
        
        except Exception as e:
            st.warning(f"Warning: Failed to process file {uploaded_file.name}: {e}")
        
        finally:
            # Delete the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    if not modified_image_masks:
        raise ValueError("No valid DICOM files uploaded")

    volume = np.stack(modified_image_masks, axis=0)
    print(volume.shape)
    volume = volume.reshape(volume.shape[0], volume.shape[2], volume.shape[3])
    print(volume.shape)
    volume = volume.transpose(2, 1, 0)
    print(volume.shape)
    return volume

# --- Streamlit UI ---
st.title("3D Lung Visualization (DICOM Upload)")

# --- DICOM File Upload ---
uploaded_files = st.file_uploader("Upload DICOM Files", type=["dcm"], accept_multiple_files=True)

if uploaded_files:
    try:
        # --- Load Lung Data ---
        lung_data = load_dicom_files(uploaded_files)

        # --- 3D Visualization ---
        verts, faces, _, _ = measure.marching_cubes(lung_data)
        mesh_plot_lung = go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            opacity=0.2, color='blue'  # Adjust color and opacity as needed
        )

        lung_fig = go.Figure(data=[mesh_plot_lung])
        lung_fig.update_layout(scene_aspectmode='data')
        st.plotly_chart(lung_fig, use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload DICOM files to visualize the lungs.")
```

**2.1  Loading and Segmenting DICOM Files: The `load_dicom_files` Function**

This function is the heart of our application. It takes a list of uploaded DICOM files, performs lung segmentation, and returns a 3D NumPy array representing the segmented lung volume.

```python
# --- Load Lung Data from Uploaded DICOM Files ---
def load_dicom_files(uploaded_files):
    """Loads a list of DICOM files and stacks the lungs after segmentation."""
    inferer = LMInferer()
    modified_image_masks = []
    for uploaded_file in uploaded_files:
        try:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_filename = temp_file.name
            
            # Read the DICOM data from the temporary file
            input_image = sitk.ReadImage(temp_filename)
            segmentation = inferer.apply(input_image)  # Apply lungmask

            # Create a mask where lungs are True, background is False
            lung_mask = segmentation != 0

            # Apply mask to the original image array
            image_array = sitk.GetArrayFromImage(input_image)
            image_array[~lung_mask] = 0  # Set background pixels to black

            modified_image_masks.append(image_array)
        
        except Exception as e:
            st.warning(f"Warning: Failed to process file {uploaded_file.name}: {e}")
        
        finally:
            # Delete the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    if not modified_image_masks:
        raise ValueError("No valid DICOM files uploaded")

    volume = np.stack(modified_image_masks, axis=0)
    print(volume.shape)
    volume = volume.reshape(volume.shape[0], volume.shape[2], volume.shape[3])
    print(volume.shape)
    volume = volume.transpose(2, 1, 0)
    print(volume.shape)
    return volume
```

**Step-by-Step Explanation:**

1.  **Initialization:**
    *   `inferer = LMInferer()` : Creates an instance of the `LMInferer` object from the `lungmask` library. This object will be used to perform automatic lung segmentation.
    *   `modified_image_masks = []` : An empty list is created to store the segmented image arrays for each DICOM file.

2.  **Iterating Through Uploaded Files:**
    *   `for uploaded_file in uploaded_files:`:  The code iterates through each DICOM file that the user uploads.

3.  **Handling DICOM Files:**
    *   **Temporary Files:**
        *   `with tempfile.NamedTemporaryFile(delete=False) as temp_file:`:  A temporary file is created to temporarily store the uploaded DICOM data. This is because Streamlit doesn't directly handle file paths.
        *   `temp_file.write(uploaded_file.getbuffer())`: Writes the DICOM data into the temporary file.
        *   `temp_filename = temp_file.name`:  Stores the temporary file's name for later access.

    *   **Reading DICOM Data:**
        *   `input_image = sitk.ReadImage(temp_filename)`:  Reads the DICOM data from the temporary file using `SimpleITK`. This creates a `SimpleITK` image object (`input_image`) that represents the DICOM data.

    *   **Lung Segmentation:**
        *   `segmentation = inferer.apply(input_image)`: Applies the lung segmentation algorithm from `lungmask` to the `input_image`. This returns a new image object (`segmentation`) where the lung regions are marked.

    *   **Creating a Lung Mask:**
        *   `lung_mask = segmentation != 0`:  Creates a boolean mask where `True` represents lung pixels and `False` represents background pixels.

    *   **Applying the Mask:**
        *   `image_array = sitk.GetArrayFromImage(input_image)`:  Converts the `input_image` (the original DICOM data) into a NumPy array (`image_array`).
        *   `image_array[~lung_mask] = 0`:  Sets the background pixels (where `lung_mask` is `False`) to 0 (black) in the NumPy array.

    *   **Adding to the List:**
        *   `modified_image_masks.append(image_array)`:  Appends the modified image array (with the lung segmentation applied) to the `modified_image_masks` list.

4.  **Error Handling:**
    *   `try-except` block:  Handles any exceptions that might occur during file processing.
    *   `st.warning(f"Warning: Failed to process file {uploaded_file.name}: {e}")`:  Displays a warning message to the user if an error occurs during file processing.
    *   `finally` block:  Ensures that the temporary file is deleted after processing, regardless of whether an exception occurred or not.

5.  **Creating the 3D Volume:**
    *   `if not modified_image_masks:`:  Checks if any valid DICOM files were uploaded.
    *   `raise ValueError("No valid DICOM files uploaded")`:  Raises an error if no valid DICOM files were uploaded.

    *   **Stacking Image Slices:**
        *   `volume = np.stack(modified_image_masks, axis=0)`:  Stacks the segmented image arrays (from the `modified_image_masks` list) along the first axis (`axis=0`) to create a 3D volume. This volume represents the 3D structure of the lungs.

    *   **Reshaping and Transposing:**
        *   `volume = volume.reshape(volume.shape[0], volume.shape[2], volume.shape[3])`: Reshapes the volume array to ensure that the dimensions are correct for visualization.
        *   `volume = volume.transpose(2, 1, 0)`: Transposes the volume array to ensure that the axes are ordered correctly for visualization.

    *   **Return the Volume:**
        *   `return volume`:  The `load_dicom_files` function returns the 3D lung volume, ready for visualization.

**2.2  Streamlit UI and Visualization**

This section of the code builds the user interface (UI) using Streamlit and generates the interactive 3D visualization.

```python
# --- Streamlit UI ---
st.title("3D Lung Visualization (DICOM Upload)")

# --- DICOM File Upload ---
uploaded_files = st.file_uploader("Upload DICOM Files", type=["dcm"], accept_multiple_files=True)

if uploaded_files:
    try:
        # --- Load Lung Data ---
        lung_data = load_dicom_files(uploaded_files)

        # --- 3D Visualization ---
        verts, faces, _, _ = measure.marching_cubes(lung_data)
        mesh_plot_lung = go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            opacity=0.2, color='blue'  # Adjust color and opacity as needed
        )

        lung_fig = go.Figure(data=[mesh_plot_lung])
        lung_fig.update_layout(scene_aspectmode='data')
        st.plotly_chart(lung_fig, use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload DICOM files to visualize the lungs.")
```

**Step-by-Step Explanation:**

1.  **Streamlit Title:**
    *   `st.title("3D Lung Visualization (DICOM Upload)")`:  Sets the title of the Streamlit application, making it clear to the user what the app does.

2.  **DICOM File Upload:**
    *   `uploaded_files = st.file_uploader("Upload DICOM Files", type=["dcm"], accept_multiple_files=True)`:  Creates a file uploader widget in the Streamlit app, allowing users to upload DICOM files. 
        *   **`type=["dcm"]`:**  Specifies that the file uploader should only accept files with the `.dcm` extension, which is the standard file extension for DICOM files.
        *   **`accept_multiple_files=True`:**  Allows the user to upload multiple DICOM files at once.

3.  **Processing Uploaded Files:**
    *   `if uploaded_files:`:  This conditional statement checks if the user has uploaded any DICOM files.
    *   **Loading Lung Data:**
        *   `lung_data = load_dicom_files(uploaded_files)`: Calls the `load_dicom_files` function, which we explained earlier, to load the DICOM files and perform lung segmentation.

    *   **3D Visualization:**
        *   **Surface Mesh Extraction:**
            *   `verts, faces, _, _ = measure.marching_cubes(lung_data)`: Uses the `marching_cubes` algorithm from `skimage` to extract the surface mesh of the segmented lung volume. The `marching_cubes` algorithm is like taking a 3D sculpture and carefully carving out the surface, creating a set of vertices (`verts`) and faces (`faces`).
        *   **Creating the Mesh Plot:**
            *   `mesh_plot_lung = go.Mesh3d(...)`: Creates a Plotly `Mesh3d` object for visualizing the 3D lung mesh. 
                *   `x=verts[:, 0], y=verts[:, 1], z=verts[:, 2]`:  Specifies the coordinates of the vertices for the 3D mesh.
                *   `i=faces[:, 0], j=faces[:, 1], k=faces[:, 2]`:  Defines the connectivity between vertices to create the faces of the mesh.
                *   `opacity=0.2`:  Sets the opacity of the mesh, allowing us to see through it to some extent.
                *   `color='blue'`:  Sets the color of the mesh (adjust this as you like).

        *   **Creating a Plotly Figure:**
            *   `lung_fig = go.Figure(data=[mesh_plot_lung])`:  Creates a Plotly figure with the 3D lung mesh as the data.
            *   `lung_fig.update_layout(scene_aspectmode='data')`:  Sets the aspect ratio of the 3D scene to match the dimensions of the lung data, ensuring that the visualization is accurate.

        *   **Displaying the Visualization:**
            *   `st.plotly_chart(lung_fig, use_container_width=True)`:  Displays the interactive 3D visualization in the Streamlit app.
                *   `use_container_width=True`:  Ensures that the visualization takes up the full width of the Streamlit app's container.

4.  **Error Handling:**
    *   `try-except` block:  Handles any exceptions that might occur during file loading, segmentation, or visualization.
    *   `st.error(f"An error occurred: {e}")`:  Displays an error message to the user if an exception occurs.

5.  **Informative Message for No Uploads:**
    *   `st.info("Please upload DICOM files to visualize the lungs.")`:  Displays a message to the user if no DICOM files have been uploaded.

### 4. Deep Dive: Lungmask and U-Net

**4.1  Lungmask: A Powerful Tool for Automatic Segmentation**

The `lungmask` library is specifically designed to automatically segment the lungs in DICOM images, making our visualization task much easier.  It's like having a skilled image editor that can automatically identify and isolate the lung regions for us!

**4.2  U-Net: The Brain Behind Lungmask**

Behind the scenes, `lungmask` relies on a powerful deep learning architecture called **U-Net.** The U-Net is a convolutional neural network (CNN) designed for image segmentation tasks. It's like a sophisticated image classifier, but instead of simply identifying objects, it creates a detailed map of where objects are located within an image.

**Think of it like this:**

*   **Image Classification:** Imagine a basic image classifier that looks at a picture and tells you whether it contains a cat or a dog.
*   **Image Segmentation:**  Imagine a more advanced image classifier that not only tells you there's a cat in the image but also outlines the cat's exact shape and location in the picture. That's what U-Net does!

**Why is U-Net so Effective for Lung Segmentation?**

*   **Encoder-Decoder Structure:**  U-Net uses a unique "encoder-decoder" structure. The encoder part of the network processes the input image, extracting important features and reducing the image's resolution. The decoder part then expands the resolution and uses the extracted features to create a detailed segmentation map.
*   **Skip Connections:**  U-Net has skip connections that connect the encoder layers to the decoder layers. These connections help the network retain important spatial information during the downsampling (encoding) process. This is crucial for accurate segmentation.
*   **Training on Medical Images:**  `Lungmask` has been trained on a large dataset of medical images, specifically focusing on lung images. This extensive training allows it to generalize well to new lung images.

**Key Benefits of using Lungmask:**

*   **Automatic Segmentation:**  `Lungmask` takes away the tedious manual segmentation work, saving time and effort.
*   **High Accuracy:**  The U-Net architecture and the extensive training data result in highly accurate lung segmentation results.
*   **Ease of Use:** `Lungmask` is designed to be user-friendly, making it easy to integrate into our Streamlit application.

### 5. Streamlit: Building an Interactive Web App

Streamlit is a Python library specifically designed for building interactive web applications for data science and machine learning. It's like a super-fast way to create web apps without needing to learn complex web development frameworks. 

**Key Features of Streamlit:**

*   **Simple Syntax:**  Streamlit uses a simple, Python-centric syntax, making it easy for data scientists to create web apps without writing a lot of HTML, CSS, or JavaScript.
*   **Rapid Prototyping:**  Streamlit is perfect for quickly prototyping ideas and creating interactive dashboards for exploring data.
*   **Interactive Widgets:**  Streamlit provides a wide range of widgets, like buttons, sliders, text boxes, file uploaders, and more, making it easy to create dynamic user interfaces.
*   **Integration with Data Science Libraries:**  Streamlit seamlessly integrates with popular data science libraries like NumPy, Pandas, Matplotlib, and Plotly, making it a natural fit for data-driven applications.

**In our application, we use Streamlit to:**

*   **Create a User Interface:**  The file uploader widget in our app, which allows users to select DICOM files, is powered by Streamlit.
*   **Display the Visualization:**  Streamlit's `st.plotly_chart` function allows us to display the interactive 3D visualization generated by Plotly.
*   **Handle User Interactions:**  Streamlit can be used to handle user inputs, such as adjusting parameters for the visualization or performing further analysis.

### 6. Running the Application: Bringing it All Together

1.  **Open your terminal** and navigate to the directory where your `main.py` file is located.
2.  **Run the Streamlit app:**
    ```bash
    streamlit run main.py
    ```
3.  **This will launch your web browser** and open the Streamlit app, which will be accessible at a URL like `http://localhost:8501/`.
4.  **Use the file uploader widget** to select one or more DICOM files.
5.  **The Streamlit app will then process the files, perform lung segmentation, and display the interactive 3D visualization of the lungs.**

### 7. Conclusion:  A World of Possibilities

You've now successfully built a 3D lung visualization app using Streamlit, `lungmask`, and Plotly.  This application demonstrates the power of Python for medical imaging and data science.

**Where to Go from Here:**

*   **Customization:**  Experiment with different visualization parameters (colors, opacity, etc.) and add user controls to the Streamlit app.
*   **Advanced Visualization:** Explore 3D rendering techniques like volume rendering or surface texture mapping for even more detailed visualization.
*   **Data Analysis:**  Integrate data analysis features into your app, such as volume calculations, abnormality detection, or comparisons between different datasets.
*   **Deployment:**  Make your app accessible to others by deploying it to a web server.

I hope this comprehensive guide has provided you with a deep understanding of 3D lung visualization, empowering you to explore the exciting world of medical imaging using Python and AI. Keep experimenting, keep learning, and keep visualizing!  
---


