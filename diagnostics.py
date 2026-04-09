
import sys
import cv2
import numpy as np
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def print_success(text):
    """Print success message"""
    print(f"SUCCESS: {text}")


def print_error(text):
    """Print error message"""
    print(f"ERROR: {text}")


def print_warning(text):
    """Print warning message"""
    print(f"WARNING:  {text}")


def check_python_version():
    """Check Python version"""
    print_header("Python Version Check")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"Python Version: {version_str}")
    
    if version.major >= 3 and version.minor >= 8:
        print_success("Python version is compatible")
        return True
    else:
        print_error("Python 3.8+ required")
        return False


def check_packages():
    """Check installed packages"""
    print_header("Required Packages Check")
    
    required = [
        "cv2",
        "numpy",
        "pandas",
        "openpyxl",
        "PIL",
    ]
    
    optional = [
        "face_recognition",
        "insightface",
        "onnxruntime",
    ]
    
    all_good = True
    
    print("\nRequired packages:")
    for package in required:
        try:
            pkg = __import__(package)
            version = getattr(pkg, '__version__', 'unknown')
            print_success(f"{package}: {version}")
        except ImportError:
            print_error(f"{package}: NOT INSTALLED")
            all_good = False
    
    print("\nOptional packages:")
    for package in optional:
        try:
            pkg = __import__(package)
            version = getattr(pkg, '__version__', 'unknown')
            print_success(f"{package}: {version}")
        except ImportError:
            print_warning(f"{package}: NOT INSTALLED (optional)")
        except Exception as e:
            print_warning(f"{package}: Error loading - {str(e)[:50]}")
    
    return all_good


def check_camera():
    print_header("Camera Check")
    
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print_success(f"Camera {i} found")
            
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps}")
            
            cap.release()
            return True
    
    print_error("No camera found")
    return False


def check_folders():

    print_header("Folder Structure Check")
    
    folders = {
        "face_data": "Student photos (must have subfolders with student names)",
        "attendance": "Attendance logs (auto-created)",
        "models": "Model files (auto-created)",
    }
    
    for folder, description in folders.items():
        path = Path(folder)
        if path.exists():
            print_success(f"{folder}/ exists")
            
            if folder == "face_data":
                subfolders = [f for f in path.iterdir() if f.is_dir()]
                if subfolders:
                    print(f"  Subfolders found: {len(subfolders)}")
                    for sub in sorted(subfolders)[:5]:
                        photo_count = len(list(sub.glob("*.jpg")))
                        print(f"    - {sub.name}: {photo_count} photos")
                else:
                    print_warning(f"  {folder}/ is empty - add student folders!")
        else:
            print_warning(f"{folder}/ not found - will be created on first run")


def check_student_photos():
    """Check student photo quality"""
    print_header("Student Photos Quality Check")
    
    face_data_path = Path("face_data")
    
    if not face_data_path.exists():
        print_warning("face_data folder not found")
        return False
    
    students = [f for f in face_data_path.iterdir() if f.is_dir()]
    
    if not students:
        print_warning("No student folders found")
        return False
    
    print(f"Found {len(students)} student(s)")
    
    all_good = True
    for student_dir in sorted(students)[:10]:  
        photos = list(student_dir.glob("*.jpg"))
        
        if len(photos) < 1:
            print_error(f"{student_dir.name}: No photos")
            all_good = False
        elif len(photos) < 3:
            print_warning(f"{student_dir.name}: Only {len(photos)} photo(s) (recommend 3+)")
        else:
            print_success(f"{student_dir.name}: {len(photos)} photos")
    
    return all_good


def check_face_recognition_library():
    """Test face_recognition library"""
    print_header("face_recognition Library Test")
    
    try:
        import face_recognition
        print_success("face_recognition imported successfully")
        
        # Test loading a model
        print("Testing face detection...")
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = face_recognition.face_locations(test_image, model="hog")
        print_success("Face detection working")
        
        return True
    except ImportError:
        print_error("face_recognition not installed")
        print("  Install with: pip install face-recognition")
        return False
    except Exception as e:
        print_error(f"Error testing face_recognition: {e}")
        return False


def check_insightface_library():
    """Test insightface library"""
    print_header("insightface Library Test")
    
    try:
        import insightface
        print_success("insightface imported successfully")
        
        print("Note: insightface models are large (~600MB)")
        print("They will be downloaded on first use")
        
        return True
    except ImportError:
        print_warning("insightface not installed (optional)")
        print("  Install with: pip install insightface")
        return False
    except Exception as e:
        print_warning(f"Error testing insightface: {e}")
        return False


def generate_recommendations():
    """Generate personalized recommendations"""
    print_header("Recommendations")
    
    print("\n1. WHICH VERSION TO USE:")
    print("   facial_recognition_v2.py")
    print("      Best for most schools (99% accuracy, fast)")
    print("\n   facial_recognition_high_accuracy.py")
    print("      For critical systems (99.9% accuracy, slower)")
    
    print("\n2. SETUP CHECKLIST:")
    print("   [ ] Install dependencies: pip install face-recognition opencv-python pandas openpyxl pillow")
    print("   [ ] Create face_data/StudentName/ folders")
    print("   [ ] Add 3-5 photos per student")
    print("   [ ] Test camera works")
    print("   [ ] Run: python facial_recognition_v2.py")
    
    print("\n3. PHOTO TIPS:")
    print("   - Good lighting (window light or office lighting)")
    print("   - Face filling ~50-80% of photo")
    print("   - Front-facing (slight angles OK)")
    print("   - JPG format, good quality")
    print("   - 3+ photos per student for best results")
    
    print("\n4. TROUBLESHOOTING:")
    print("   - Not recognizing: Add more photos, improve lighting")
    print("   - Too many false positives: Lower sensitivity slider")
    print("   - Slow processing: Use 'hog' model instead of 'cnn'")
    print("   - Camera not working: Check Device Manager, try different USB port")
    
    print("\n5. READ GUIDES:")
    print("   - SCHOOL_SETUP_GUIDE.md (how to use for school)")
    print("   - SYSTEM_COMPARISON.md (compare versions)")


def main():
    """Run all diagnostics"""
    print("\n" + "█"*60)
    print("█  Facial Recognition System - Diagnostics")
    print("█  Run this to verify your setup is correct")
    print("█"*60)
    
    results = {}
    
  
    results['python'] = check_python_version()
    results['packages'] = check_packages()
    results['camera'] = check_camera()
    results['folders'] = check_folders()
    results['photos'] = check_student_photos()
    results['face_recognition'] = check_face_recognition_library()
    results['insightface'] = check_insightface_library()
    

    print_header("Summary")
    
    if results['python'] and results['packages'] and results['camera']:
        print_success("All critical checks passed!")
        print("\nYou're ready to run the system:")
        print("  python facial_recognition_v2.py")
    else:
        print_error("Some critical checks failed")
        print("\nPlease fix the issues above and try again")
    

    generate_recommendations()
    
    print("\n" + "█"*60 + "\n")


if __name__ == "__main__":
    main()
