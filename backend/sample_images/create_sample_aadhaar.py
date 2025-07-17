from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_aadhaar():
    """Create a sample Aadhaar card image for testing."""
    
    # Create a new image with white background
    width, height = 800, 500
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fall back to default if not available
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        normal_font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except:
        title_font = ImageFont.load_default()
        normal_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw header
    draw.text((50, 30), "भारत सरकार / Government of India", fill='blue', font=title_font)
    draw.text((50, 60), "आधार / Aadhaar", fill='blue', font=title_font)
    
    # Draw sample data
    y_position = 120
    
    # Name
    draw.text((50, y_position), "Name / नाम:", fill='black', font=normal_font)
    draw.text((200, y_position), "RAJ KUMAR SHARMA", fill='black', font=normal_font)
    y_position += 30
    
    # Aadhaar number
    draw.text((50, y_position), "Aadhaar Number / आधार संख्या:", fill='black', font=normal_font)
    draw.text((200, y_position), "1234 5678 9012", fill='black', font=normal_font)
    y_position += 30
    
    # Date of Birth
    draw.text((50, y_position), "Date of Birth / जन्म तिथि:", fill='black', font=normal_font)
    draw.text((200, y_position), "15/03/1990", fill='black', font=normal_font)
    y_position += 30
    
    # Gender
    draw.text((50, y_position), "Gender / लिंग:", fill='black', font=normal_font)
    draw.text((200, y_position), "Male / पुरुष", fill='black', font=normal_font)
    y_position += 30
    
    # Address
    draw.text((50, y_position), "Address / पता:", fill='black', font=normal_font)
    y_position += 25
    draw.text((200, y_position), "Flat No. 123, Building A", fill='black', font=normal_font)
    y_position += 25
    draw.text((200, y_position), "Sector 15, Dwarka", fill='black', font=normal_font)
    y_position += 25
    draw.text((200, y_position), "New Delhi - 110075", fill='black', font=normal_font)
    y_position += 30
    
    # Phone number
    draw.text((50, y_position), "Phone / फोन:", fill='black', font=normal_font)
    draw.text((200, y_position), "+91 98765 43210", fill='black', font=normal_font)
    y_position += 30
    
    # Email
    draw.text((50, y_position), "Email / ईमेल:", fill='black', font=normal_font)
    draw.text((200, y_position), "raj.sharma@email.com", fill='black', font=normal_font)
    y_position += 30
    
    # PAN number
    draw.text((50, y_position), "PAN / पैन:", fill='black', font=normal_font)
    draw.text((200, y_position), "ABCDE1234F", fill='black', font=normal_font)
    
    # Draw border
    draw.rectangle([(20, 20), (width-20, height-20)], outline='blue', width=2)
    
    # Add some decorative elements
    draw.rectangle([(50, 100), (width-50, 110)], fill='blue')
    
    # Save the image
    os.makedirs('sample_images', exist_ok=True)
    image.save('sample_images/sample_aadhaar.jpg', 'JPEG', quality=95)
    print("Sample Aadhaar card created: sample_images/sample_aadhaar.jpg")

if __name__ == "__main__":
    create_sample_aadhaar() 