import json
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from datetime import datetime
from selenium.webdriver.common.keys import Keys

class SemanticFormFiller:
    def __init__(self):
        # Initialize the sentence transformer model
        print("Loading semantic model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully!")
        
        # Load user data
        with open('user_data.json', 'r') as f:
            self.form_data = json.load(f)
        
        # Create embeddings for all field names
        self.field_embeddings = self.create_field_embeddings()
        
    def create_field_embeddings(self):
        """Create embeddings for all field names in the form data"""
        field_names = list(self.form_data.keys())
        embeddings = self.model.encode(field_names, convert_to_tensor=True)
        return dict(zip(field_names, embeddings))
    
    def find_best_match(self, question_text, threshold=0.5):
        """Find the best matching field name for a given question"""
        # Encode the question
        question_embedding = self.model.encode(question_text, convert_to_tensor=True)
        
        # Calculate similarities with all field names
        similarities = {}
        for field_name, field_embedding in self.field_embeddings.items():
            similarity = util.pytorch_cos_sim(question_embedding, field_embedding).item()
            similarities[field_name] = similarity
        
        # Find the best match
        best_match = max(similarities.items(), key=lambda x: x[1])
        
        # Return the match if it's above the threshold
        if best_match[1] >= threshold:
            return best_match[0]
        return None
    
    def setup_driver(self):
        """Set up Chrome WebDriver"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--start-maximized')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            
            service = Service()
            driver = webdriver.Chrome(service=service, options=chrome_options)
            return driver
            
        except Exception as e:
            print(f"Error setting up Chrome driver: {str(e)}")
            print("Please make sure Chrome is installed on your system.")
            raise
    
    def get_field_type(self, element):
        """Determine the type of form field"""
        try:
            # Check for multiple choice
            if element.find_elements(By.XPATH, ".//div[@role='radio']"):
                return "radio"
            # Check for checkbox
            elif element.find_elements(By.XPATH, ".//div[@role='checkbox']"):
                return "checkbox"
            # Check for date picker
            elif element.find_elements(By.XPATH, ".//input[@type='date']"):
                return "date"
            # Check for dropdown
            elif element.find_elements(By.XPATH, ".//div[@role='listbox']"):
                return "dropdown"
            # Check for textarea
            elif element.find_elements(By.XPATH, ".//textarea"):
                return "textarea"
            # Check for text input
            elif element.find_elements(By.XPATH, ".//input[@type='text']"):
                return "text"
            # Default to text input
            else:
                return "text"
        except:
            return "text"
    
    def fill_date_field(self, driver, element, date_str):
        """Fill a date field in Google Forms"""
        try:
            # Parse the date string (assuming format: MM/DD/YYYY)
            date_obj = datetime.strptime(date_str, "%m/%d/%Y")
            
            # Find the date input field
            date_input = element.find_element(By.XPATH, ".//input[@type='date']")
            
            # Format date as YYYY-MM-DD for the input field
            formatted_date = date_obj.strftime("%Y-%m-%d")
            
            # Clear any existing value
            date_input.clear()
            
            # Try to set the date directly first
            try:
                date_input.send_keys(formatted_date)
                return True
            except:
                pass
            
            # If direct input fails, try clicking and using keyboard
            try:
                # Click the input field
                date_input.click()
                time.sleep(0.5)
                
                # Send the date parts separately
                date_input.send_keys(str(date_obj.year))
                date_input.send_keys(Keys.TAB)
                date_input.send_keys(str(date_obj.month).zfill(2))
                date_input.send_keys(Keys.TAB)
                date_input.send_keys(str(date_obj.day).zfill(2))
                return True
            except:
                pass
            
            # If keyboard input fails, try JavaScript
            try:
                driver.execute_script(
                    "arguments[0].value = arguments[1];", 
                    date_input, 
                    formatted_date
                )
                return True
            except:
                pass
            
            print(f"Warning: Could not set date {date_str} using any method")
            return False
            
        except Exception as e:
            print(f"Error filling date field: {str(e)}")
            return False
    
    def fill_radio_field(self, driver, element, value):
        """Fill a radio button field"""
        try:
            # Find all radio options
            radio_options = element.find_elements(By.XPATH, ".//div[@role='radio']")
            
            # First try exact match
            for option in radio_options:
                option_text = option.text.strip().lower()
                if value.lower() == option_text:
                    option.click()
                    return True
            
            # Then try partial match
            for option in radio_options:
                option_text = option.text.strip().lower()
                if value.lower() in option_text or option_text in value.lower():
                    option.click()
                    return True
            
            # If no direct match, try semantic matching
            for option in radio_options:
                option_text = option.text.strip()
                similarity = util.pytorch_cos_sim(
                    self.model.encode(option_text, convert_to_tensor=True),
                    self.model.encode(value, convert_to_tensor=True)
                ).item()
                
                if similarity > 0.7:  # High similarity threshold for radio buttons
                    option.click()
                    return True
            
            # If still no match, try to find the closest option
            best_match = None
            best_similarity = 0
            
            for option in radio_options:
                option_text = option.text.strip()
                similarity = util.pytorch_cos_sim(
                    self.model.encode(option_text, convert_to_tensor=True),
                    self.model.encode(value, convert_to_tensor=True)
                ).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = option
            
            if best_match and best_similarity > 0.5:  # Lower threshold for best match
                best_match.click()
                return True
            
            return False
        except Exception as e:
            print(f"Error filling radio field: {str(e)}")
            return False
    
    def fill_dropdown_field(self, driver, element, value):
        """Fill a dropdown field"""
        try:
            # Click the dropdown to open it
            dropdown = element.find_element(By.XPATH, ".//div[@role='listbox']")
            dropdown.click()
            time.sleep(0.5)
            
            # Find all options
            options = driver.find_elements(By.XPATH, "//div[@role='option']")
            
            # First try exact match
            for option in options:
                option_text = option.text.strip().lower()
                if value.lower() == option_text:
                    option.click()
                    return True
            
            # Then try partial match
            for option in options:
                option_text = option.text.strip().lower()
                if value.lower() in option_text or option_text in value.lower():
                    option.click()
                    return True
            
            # If no direct match, try semantic matching
            for option in options:
                option_text = option.text.strip()
                similarity = util.pytorch_cos_sim(
                    self.model.encode(option_text, convert_to_tensor=True),
                    self.model.encode(value, convert_to_tensor=True)
                ).item()
                
                if similarity > 0.7:  # High similarity threshold for dropdowns
                    option.click()
                    return True
            
            # If still no match, try to find the closest option
            best_match = None
            best_similarity = 0
            
            for option in options:
                option_text = option.text.strip()
                similarity = util.pytorch_cos_sim(
                    self.model.encode(option_text, convert_to_tensor=True),
                    self.model.encode(value, convert_to_tensor=True)
                ).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = option
            
            if best_match and best_similarity > 0.5:  # Lower threshold for best match
                best_match.click()
                return True
            
            return False
        except Exception as e:
            print(f"Error filling dropdown field: {str(e)}")
            return False
    
    def fill_text_field(self, driver, element, value):
        """Fill a text input field"""
        try:
            # Try different input field types
            input_types = [
                ".//input[@type='text']",
                ".//input",
                ".//textarea"
            ]
            
            for input_type in input_types:
                try:
                    input_field = element.find_element(By.XPATH, input_type)
                    input_field.clear()
                    input_field.send_keys(str(value))
                    return True
                except NoSuchElementException:
                    continue
            
            return False
        except Exception as e:
            print(f"Error filling text field: {str(e)}")
            return False
    
    def find_field_by_text(self, driver, text):
        """Find a form field by its label text"""
        try:
            # Try to find the question container
            xpath_patterns = [
                f"//div[contains(@class, 'Qr7Oae')]//span[contains(text(), '{text}')]",
                f"//div[contains(@class, 'geS5n')]//span[contains(text(), '{text}')]",
                f"//div[contains(@class, 'M7eMe')]//span[contains(text(), '{text}')]",
                f"//div[contains(@class, 'freebirdFormviewerViewItemsItemItem')]//span[contains(text(), '{text}')]"
            ]
            
            for pattern in xpath_patterns:
                try:
                    element = driver.find_element(By.XPATH, pattern)
                    # Get the parent container
                    container = element.find_element(By.XPATH, ".//ancestor::div[contains(@class, 'geS5n')]")
                    return container
                except NoSuchElementException:
                    continue
            
            return None
        except Exception as e:
            print(f"Error finding field by text '{text}': {str(e)}")
            return None
    
    def get_form_questions(self, driver):
        """Extract all questions from the form"""
        questions = []
        try:
            # Find all question elements
            question_patterns = [
                "//div[contains(@class, 'Qr7Oae')]//span[contains(@class, 'M7eMe')]",
                "//div[contains(@class, 'freebirdFormviewerViewItemsItemItem')]//span[contains(@class, 'M7eMe')]",
                "//div[contains(@class, 'geS5n')]//span[contains(@class, 'M7eMe')]"
            ]
            
            for pattern in question_patterns:
                elements = driver.find_elements(By.XPATH, pattern)
                for element in elements:
                    question_text = element.text.strip()
                    if question_text and question_text not in questions:
                        questions.append(question_text)
            
            return questions
        except Exception as e:
            print(f"Error extracting questions: {str(e)}")
            return []
    
    def fill_form(self, driver):
        """Fill the form using semantic matching"""
        try:
            # Wait for form to load
            print("Waiting for form to load...")
            time.sleep(3)
            
            # Get all questions from the form
            questions = self.get_form_questions(driver)
            print(f"\nFound {len(questions)} questions in the form")
            
            # Process each question
            for question in questions:
                print(f"\nProcessing question: {question}")
                
                # Find the best matching field
                best_match = self.find_best_match(question)
                
                if best_match:
                    print(f"Matched with field: {best_match}")
                    value = self.form_data[best_match]
                    
                    # Find the form field container
                    field_container = self.find_field_by_text(driver, question)
                    
                    if field_container:
                        # Determine field type and fill accordingly
                        field_type = self.get_field_type(field_container)
                        print(f"Field type: {field_type}")
                        
                        # Scroll to the element
                        driver.execute_script("arguments[0].scrollIntoView(true);", field_container)
                        time.sleep(0.5)
                        
                        # Fill based on field type
                        if field_type == "date":
                            success = self.fill_date_field(driver, field_container, value)
                        elif field_type == "radio":
                            success = self.fill_radio_field(driver, field_container, value)
                        elif field_type == "dropdown":
                            success = self.fill_dropdown_field(driver, field_container, value)
                        else:
                            success = self.fill_text_field(driver, field_container, value)
                        
                        if success:
                            print(f"Successfully filled with: {value}")
                        else:
                            print(f"Failed to fill field with value: {value}")
                        time.sleep(0.5)
                    else:
                        print(f"Could not find input field for question: {question}")
                else:
                    print(f"No matching field found for question: {question}")
            
            print("\nForm filling completed!")
            print("Please review the filled form and submit it manually.")
            print("Press Enter when you're done...")
            input()
            
        except Exception as e:
            print(f"Error filling form: {str(e)}")
        finally:
            time.sleep(2)
    
    def run(self):
        """Run the form filler"""
        # Get form URL from user
        form_url = input("Please enter the form URL: ").strip()
        
        # Setup driver
        driver = self.setup_driver()
        
        try:
            # Navigate to form URL
            print(f"\nNavigating to form URL: {form_url}")
            driver.get(form_url)
            
            # Fill the form
            self.fill_form(driver)
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            input("Press Enter to close the browser...")
            driver.quit()

if __name__ == "__main__":
    # Create and run the form filler
    form_filler = SemanticFormFiller()
    form_filler.run() 