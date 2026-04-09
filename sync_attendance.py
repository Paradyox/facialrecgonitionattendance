"""
Simple script to sync attendance data for the dashboard
Reads from face_data folders and today's Excel file
Can be run independently or automatically
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd

def export_attendance_data():
    """Export attendance data for dashboard"""
    try:
        # Paths
        data_dir = Path("face_data")
        attendance_dir = Path("attendance")
        
        print(f"Looking for students in: {data_dir.absolute()}")
        print(f"Looking for attendance files in: {attendance_dir.absolute()}")
        
        # Get ALL students from face_data folders
        all_students = []
        if data_dir.exists():
            for person_dir in sorted(data_dir.iterdir()):
                if person_dir.is_dir():
                    all_students.append(person_dir.name)
                    print(f"  Found student: {person_dir.name}")
        else:
            print(f"  ERROR: face_data folder not found!")
            return False
        
        if not all_students:
            print("  WARNING: No students found in face_data!")
        else:
            print(f"  Total students found: {len(all_students)}")
        
        # Read today's attendance from Excel
        present_students = set()
        date_str = datetime.now().strftime("%Y%m%d")
        excel_path = attendance_dir / f"attendance_{date_str}.xlsx"
        
        print(f"\nLooking for today's attendance file: {excel_path}")
        
        if excel_path.exists():
            try:
                print(f"  Found file: {excel_path}")
                df = pd.read_excel(excel_path)
                print(f"  Excel file has {len(df)} entries")
                
                # Handle different column names
                person_col = None
                status_col = None
                
                for col in df.columns:
                    col_lower = col.lower()
                    if 'person' in col_lower or 'name' in col_lower:
                        person_col = col
                    if 'status' in col_lower:
                        status_col = col
                
                if not person_col or not status_col:
                    print(f"  ERROR: Expected 'Person' and 'Status' columns, found: {list(df.columns)}")
                    return False
                
                for _, row in df.iterrows():
                    person = str(row[person_col]).strip()
                    status = str(row[status_col]).strip()
                    print(f"    {person}: {status}")
                    if status.lower() == 'present':
                        present_students.add(person)
                
                print(f"  Total present: {len(present_students)}")
            except Exception as e:
                print(f"  ERROR reading Excel file: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"  WARNING: File not found. Checking if any attendance files exist...")
            # List all attendance files
            if attendance_dir.exists():
                files = list(attendance_dir.glob("*.xlsx"))
                if files:
                    print(f"  Found {len(files)} attendance files:")
                    for f in files:
                        print(f"    {f.name}")
                else:
                    print(f"  ERROR: No attendance files found in {attendance_dir}")
                    return False
            else:
                print(f"  ERROR: Attendance folder not found!")
                return False
        
        # Build student data
        student_data = []
        for student_name in all_students:
            student_data.append({
                'name': student_name,
                'present': student_name in present_students
            })
        
        # Write to JSON
        json_path = Path('attendance_live.json')
        with open(json_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'date': datetime.now().strftime("%Y-%m-%d"),
                'students': student_data
            }, f, indent=2)
        
        print(f"\n✓ SUCCESS: Exported {len(student_data)} students ({len(present_students)} present)")
        print(f"  Saved to: {json_path.absolute()}")
        return True
        
    except Exception as e:
        print(f"ERROR exporting attendance: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    export_attendance_data()

