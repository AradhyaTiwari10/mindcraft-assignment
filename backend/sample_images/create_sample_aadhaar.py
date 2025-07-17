from PIL import Image, ImageDraw, ImageFont
import os

print("[DEBUG] Starting Aadhaar sample image creation...")

def create_sample_aadhaar():
    """Create a sample Aadhaar card image for testing."""
    
    # Create a new image with white background
    width, height = 800, 500
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to load fonts, with fallbacks for Unicode support
    try:
        # Try to find a Unicode-compatible font
        font_paths = [
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:/Windows/Fonts/arial.ttf",  # Windows
            "/System/Library/Fonts/Helvetica.ttc",  # macOS alternative
        ]
        
        title_font = None
        normal_font = None
        small_font = None
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    title_font = ImageFont.truetype(font_path, 24)
                    normal_font = ImageFont.truetype(font_path, 16)
                    small_font = ImageFont.truetype(font_path, 12)
                    print(f"[DEBUG] Using font: {font_path}")
                    break
                except:
                    continue
        
        # If no system font found, use default
        if title_font is None:
            title_font = ImageFont.load_default()
            normal_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            print("[DEBUG] Using default font")
            
    except Exception as e:
        print(f"[DEBUG] Font loading error: {e}")
        title_font = ImageFont.load_default()
        normal_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw header (use English only to avoid Unicode issues)
    draw.text((50, 30), "Government of India", fill='blue', font=title_font)
    draw.text((50, 60), "Aadhaar", fill='blue', font=title_font)
    
    # Draw sample data
    y_position = 120
    
    # Name
    draw.text((50, y_position), "Name:", fill='black', font=normal_font)
    draw.text((200, y_position), "RAJ KUMAR SHARMA", fill='black', font=normal_font)
    y_position += 30
    
    # Aadhaar number
    draw.text((50, y_position), "Aadhaar Number:", fill='black', font=normal_font)
    draw.text((200, y_position), "1234 5678 9012", fill='black', font=normal_font)
    y_position += 30
    
    # Date of Birth
    draw.text((50, y_position), "Date of Birth:", fill='black', font=normal_font)
    draw.text((200, y_position), "15/03/1990", fill='black', font=normal_font)
    y_position += 30
    
    # Gender
    draw.text((50, y_position), "Gender:", fill='black', font=normal_font)
    draw.text((200, y_position), "Male", fill='black', font=normal_font)
    y_position += 30
    
    # Address
    draw.text((50, y_position), "Address:", fill='black', font=normal_font)
    y_position += 25
    draw.text((200, y_position), "Flat No. 123, Building A", fill='black', font=normal_font)
    y_position += 25
    draw.text((200, y_position), "Sector 15, Dwarka", fill='black', font=normal_font)
    y_position += 25
    draw.text((200, y_position), "New Delhi - 110075", fill='black', font=normal_font)
    y_position += 30
    
    # Phone number
    draw.text((50, y_position), "Phone:", fill='black', font=normal_font)
    draw.text((200, y_position), "+91 98765 43210", fill='black', font=normal_font)
    y_position += 30
    
    # Email
    draw.text((50, y_position), "Email:", fill='black', font=normal_font)
    draw.text((200, y_position), "raj.sharma@email.com", fill='black', font=normal_font)
    y_position += 30
    
    # PAN number
    draw.text((50, y_position), "PAN:", fill='black', font=normal_font)
    draw.text((200, y_position), "ABCDE1234F", fill='black', font=normal_font)
    
    # Draw border
    draw.rectangle([(20, 20), (width-20, height-20)], outline='blue', width=2)
    
    # Add some decorative elements
    draw.rectangle([(50, 100), (width-50, 110)], fill='blue')
    
    # Save the image
    os.makedirs('sample_images', exist_ok=True)
    print("[DEBUG] About to save image to sample_images/sample_aadhaar.jpg")
    image.save('sample_images/sample_aadhaar.jpg', 'JPEG', quality=95)
    print("Sample Aadhaar card created: sample_images/sample_aadhaar.jpg")

if __name__ == "__main__":
    create_sample_aadhaar() 