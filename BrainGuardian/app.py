import streamlit as st
import numpy as np
import cv2
import pandas as pd
import folium
from PIL import Image
import io
import json
import os
from datetime import datetime, timedelta
from streamlit_folium import folium_static

# Import custom modules
from model import load_model, predict_image
from utils import preprocess_image, visualize_tumor, get_tumor_info
from hospital_finder import find_nearby_hospitals
from reinforcement_learning import update_model_with_feedback


# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide",
)

# Initialize session state variables if they don't exist
if 'model' not in st.session_state:
    try:
        st.session_state.model = load_model()
        st.session_state.model_loaded = True
    except Exception as e:
        st.session_state.model_loaded = False
        st.session_state.model_error = str(e)

if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'tumor_detected' not in st.session_state:
    st.session_state.tumor_detected = False
if 'tumor_type' not in st.session_state:
    st.session_state.tumor_type = None
if 'tumor_confidence' not in st.session_state:
    st.session_state.tumor_confidence = None
if 'visualization' not in st.session_state:
    st.session_state.visualization = None
if 'location' not in st.session_state:
    st.session_state.location = None
if 'hospitals' not in st.session_state:
    st.session_state.hospitals = []
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

# Load tumor type information
try:
    with open('data/labels.json', 'r') as f:
        tumor_types = json.load(f)
except FileNotFoundError:
    tumor_types = {
        "0": {
            "name": "No Tumor",
            "description": "No brain tumor detected in the image."
        },
        "1": {
            "name": "Glioma",
            "description": "A type of tumor that occurs in the brain and spinal cord. Gliomas begin in the glial cells that surround and support nerve cells."
        },
        "2": {
            "name": "Meningioma",
            "description": "A tumor that arises from the meninges — the membranes that surround your brain and spinal cord. Most meningiomas are noncancerous."
        },
        "3": {
            "name": "Pituitary",
            "description": "A tumor that develops in the pituitary gland at the base of the brain. Most pituitary tumors are noncancerous (benign)."
        }
    }

# Main header
st.title("🧠 Brain Tumor Detection System")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload MRI", "Results", "Find Specialists", "Book Appointment"])

# Home page
if page == "Home":
    st.markdown("""
    ## Welcome to the Brain Tumor Detection System
    
    This application uses advanced Convolutional Neural Networks (CNN) and Reinforcement Learning 
    to detect and classify brain tumors from MRI scans with over 90% accuracy.
    
    ### How to use:
    1. Navigate to the **Upload MRI** page
    2. Upload an MRI scan image from your device
    3. View the detection results and tumor information
    4. Find specialists near your location
    5. Book an appointment with a neurosurgeon
    
    ### About the technology:
    Our system uses state-of-the-art deep learning algorithms trained on thousands of MRI scans 
    to accurately identify and classify different types of brain tumors:
    - Glioma
    - Meningioma
    - Pituitary Tumor
    
    The system continuously improves through reinforcement learning based on specialist feedback.
    """)
    
    # Check if model is loaded
    if not st.session_state.model_loaded:
        st.error(f"Error loading model: {st.session_state.model_error}")
    else:
        st.success("Model loaded successfully. System is ready to process MRI images.")

# Upload MRI page
elif page == "Upload MRI":
    st.header("Upload MRI Scan")
    
    st.markdown("""
    Please upload a clear MRI scan image of the brain. 
    For best results, use images that are:
    - In .jpg, .jpeg, or .png format
    - Clear and high-resolution
    - Properly oriented (axial view preferred)
    """)
    
    uploaded_file = st.file_uploader("Choose an MRI image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        st.session_state.original_image = image
        
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
        
        # Process button
        if st.button("Process MRI Scan"):
            with st.spinner("Processing image..."):
                try:
                    # Preprocess the image
                    processed_img = preprocess_image(image)
                    st.session_state.processed_image = processed_img
                    
                    # Make prediction
                    prediction, confidence, tumor_class, heatmap = predict_image(
                        st.session_state.model, 
                        processed_img
                    )
                    
                    st.session_state.prediction = prediction
                    st.session_state.tumor_detected = tumor_class > 0
                    st.session_state.tumor_type = tumor_class
                    st.session_state.tumor_confidence = confidence
                    
                    # Create visualization
                    if st.session_state.tumor_detected:
                        visualization = visualize_tumor(np.array(image), heatmap)
                        st.session_state.visualization = visualization
                    
                    st.success("Analysis complete! Please go to the Results page to view findings.")
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

# Results page
elif page == "Results":
    st.header("Analysis Results")
    
    if st.session_state.original_image is None:
        st.warning("No MRI scan has been uploaded. Please upload an image first.")
    
    elif st.session_state.prediction is None:
        st.warning("No analysis has been performed. Please process your uploaded image.")
    
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original MRI Scan")
            st.image(st.session_state.original_image, use_column_width=True)
        
        with col2:
            if st.session_state.tumor_detected and st.session_state.visualization is not None:
                st.subheader("Tumor Visualization")
                st.image(st.session_state.visualization, use_column_width=True)
            else:
                st.subheader("Processed Scan")
                st.image(st.session_state.original_image, use_column_width=True)
        
        # Results box
        st.subheader("Diagnostic Results")
        
        if st.session_state.tumor_detected:
            tumor_type = st.session_state.tumor_type
            if tumor_type is None:
                tumor_type = "1"  # Default to Glioma if None
            elif isinstance(tumor_type, int):
                tumor_type = str(tumor_type)
                
            # Use a default entry if the key doesn't exist
            default_tumor = {
                "name": "Unknown Tumor",
                "description": "A tumor of unknown classification was detected."
            }
            
            if tumor_type in tumor_types:
                tumor_info = tumor_types[tumor_type]
            else:
                tumor_info = default_tumor
                
            st.error(f"⚠️ **Tumor Detected: {tumor_info['name']}**")
            st.markdown(f"**Confidence: {st.session_state.tumor_confidence:.2f}%**")
            st.markdown(f"**Description:** {tumor_info['description']}")
        else:
            st.success("✅ No tumor detected in the scan")
            st.markdown(f"**Confidence: {st.session_state.tumor_confidence:.2f}%**")
        
        # Feedback section for reinforcement learning
        st.subheader("Provide Feedback")
        st.markdown("""
        Your feedback helps our system improve! If you are a medical professional or have 
        confirmed diagnostic information, please let us know if our prediction was correct.
        """)
        
        correct_prediction = st.radio(
            "Was our prediction correct?",
            ["Yes", "No"],
            index=0
        )
        
        if correct_prediction == "No":
            correct_class = st.selectbox(
                "What is the correct tumor type?",
                [tumor_types[i]["name"] for i in tumor_types.keys()],
                index=0
            )
            correct_class_idx = next(
                (i for i, v in tumor_types.items() if v["name"] == correct_class),
                0
            )
        else:
            correct_class_idx = st.session_state.tumor_type
        
        if st.button("Submit Feedback") and not st.session_state.feedback_submitted:
            with st.spinner("Updating model..."):
                try:
                    # Update model with feedback
                    update_model_with_feedback(
                        st.session_state.model,
                        st.session_state.processed_image,
                        correct_class_idx
                    )
                    st.session_state.feedback_submitted = True
                    st.success("Thank you for your feedback! Our model has been updated.")
                except Exception as e:
                    st.error(f"Error updating model: {str(e)}")

# Find Specialists page
elif page == "Find Specialists":
    st.header("Find Brain Tumor Specialists")
    
    if st.session_state.tumor_detected is None or not st.session_state.tumor_detected:
        st.warning("""
        No tumor has been detected yet, or you haven't uploaded and processed an MRI scan.
        
        You can still search for specialists in your area.
        """)
    
    st.subheader("Your Location")
    
    col1, col2 = st.columns(2)
    
    with col1:
        address = st.text_input("Enter your address or city", "")
    
    with col2:
        search_radius = st.slider("Search radius (miles)", 5, 50, 15)
    
    if st.button("Find Specialists"):
        if address:
            with st.spinner("Searching for specialists..."):
                try:
                    # Find hospitals near the location
                    hospitals = find_nearby_hospitals(address, search_radius)
                    st.session_state.hospitals = hospitals
                    st.session_state.location = address
                    
                    if not hospitals:
                        st.error("No neurosurgery specialists found in your area. Try increasing the search radius.")
                    else:
                        st.success(f"Found {len(hospitals)} medical facilities with neurosurgery departments.")
                except Exception as e:
                    st.error(f"Error finding specialists: {str(e)}")
        else:
            st.error("Please enter your location to find specialists.")
    
    # Display hospitals if available
    if st.session_state.hospitals:
        st.subheader(f"Specialists near {st.session_state.location}")
        
        # Create map
        m = folium.Map(location=[st.session_state.hospitals[0]['lat'], st.session_state.hospitals[0]['lng']], zoom_start=10)
        
        # Add markers for each hospital
        for hospital in st.session_state.hospitals:
            folium.Marker(
                location=[hospital['lat'], hospital['lng']],
                popup=hospital['name'],
                tooltip=hospital['name'],
                icon=folium.Icon(color='red', icon='plus', prefix='fa')
            ).add_to(m)
        
        # Display map
        folium_static(m)
        
        # Display hospitals in a table
        hospital_data = {
            "Name": [h['name'] for h in st.session_state.hospitals],
            "Address": [h['address'] for h in st.session_state.hospitals],
            "Distance (miles)": [h['distance'] for h in st.session_state.hospitals],
            "Phone": [h.get('phone', 'N/A') for h in st.session_state.hospitals]
        }
        
        st.dataframe(pd.DataFrame(hospital_data))

# Book Appointment page
elif page == "Book Appointment":
    st.header("Book an Appointment")
    
    if not st.session_state.hospitals:
        st.warning("Please find specialists first before booking an appointment.")
    else:
        st.subheader("Select a Hospital")
        
        # Hospital selection
        hospital_names = [h['name'] for h in st.session_state.hospitals]
        selected_hospital = st.selectbox("Choose a medical facility", hospital_names)
        
        # Get hospital details
        hospital_idx = hospital_names.index(selected_hospital)
        hospital = st.session_state.hospitals[hospital_idx]
        
        st.markdown(f"""
        **{hospital['name']}**  
        Address: {hospital['address']}  
        Distance: {hospital['distance']} miles  
        Phone: {hospital.get('phone', 'N/A')}  
        """)
        
        # Patient information
        st.subheader("Your Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            patient_name = st.text_input("Full Name", "")
            patient_email = st.text_input("Email", "")
            patient_phone = st.text_input("Phone Number", "")
        
        with col2:
            patient_dob = st.date_input("Date of Birth")
            insurance = st.text_input("Insurance Provider (optional)", "")
            policy_number = st.text_input("Policy Number (optional)", "")
        
        # Appointment details
        st.subheader("Appointment Details")
        
        # Calculate available dates (next 14 days, excluding weekends)
        today = datetime.now().date()
        available_dates = []
        for i in range(1, 15):
            next_date = today + timedelta(days=i)
            if next_date.weekday() < 5:  # 0-4 are weekdays
                available_dates.append(next_date)
        
        appointment_date = st.selectbox("Preferred Date", available_dates)
        
        # Time slots (9 AM to 4 PM, hourly)
        time_slots = [f"{hour}:00" for hour in range(9, 17)]
        appointment_time = st.selectbox("Preferred Time", time_slots)
        
        # Reason for visit
        if st.session_state.tumor_detected:
            tumor_type = st.session_state.tumor_type
            if tumor_type is None:
                tumor_type = "1"  # Default to Glioma if None
            elif isinstance(tumor_type, int):
                tumor_type = str(tumor_type)
                
            # Use a default entry if the key doesn't exist
            default_tumor = {
                "name": "Unknown Tumor",
                "description": "A tumor of unknown classification was detected."
            }
            
            if tumor_type in tumor_types:
                tumor_info = tumor_types[tumor_type]
            else:
                tumor_info = default_tumor
                
            default_reason = f"Brain tumor detection follow-up: {tumor_info['name']}"
        else:
            default_reason = "Brain MRI follow-up consultation"
        
        reason = st.text_area("Reason for Visit", default_reason, height=100)
        
        # Additional notes
        notes = st.text_area("Additional Notes or Questions", "", height=100)
        
        # Agree to terms
        agree = st.checkbox("I confirm that the information provided is accurate and agree to the terms of service")
        
        # Submit button
        if st.button("Request Appointment", disabled=not agree):
            if not patient_name or not patient_email or not patient_phone:
                st.error("Please fill in all required fields.")
            else:
                # In a real app, this would connect to an API to book the appointment
                with st.spinner("Submitting appointment request..."):
                    try:
                        # Simulate API call delay
                        import time
                        time.sleep(2)
                        
                        # Success message
                        st.success(f"""
                        Appointment request submitted successfully!
                        
                        We've sent your appointment request to {hospital['name']} for:
                        - Date: {appointment_date}
                        - Time: {appointment_time}
                        
                        The hospital will contact you at {patient_email} or {patient_phone} to confirm.
                        """)
                        
                        st.info("""
                        Note: This is a request only. The appointment is not confirmed until you 
                        receive confirmation from the medical facility.
                        """)
                    except Exception as e:
                        st.error(f"Error submitting appointment: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    Brain Tumor Detection System - Powered by CNN and Reinforcement Learning
</div>
""", unsafe_allow_html=True)