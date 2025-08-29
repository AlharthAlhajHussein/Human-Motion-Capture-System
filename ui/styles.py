# styles.py - Comprehensive UI Stylesheet for 3D Human Motion Capture System

class UIStyles:
    """
    Centralized styling class for the 3D Human Motion Capture System UI.
    Contains all styling definitions with a modern, professional appearance.
    """
    
    # Color Palette
    COLORS = {
        # Primary Colors
        'primary_dark': '#1a1a2e',
        'primary_medium': '#16213e',
        'primary_light': '#0f4c75',
        'accent_blue': '#3282b8',
        'accent_cyan': '#00d4ff',
        
        # UI Colors
        'background_main': '#f0f2f5',
        'background_dark': '#2c3e50',
        'background_card': '#ffffff',
        'background_input': '#f8f9fa',
        'border_light': '#e1e8ed',
        'border_medium': '#cbd5e0',
        
        # Status Colors
        'success': '#27ae60',
        'warning': '#f39c12',
        'error': '#e74c3c',
        'info': '#3498db',
        
        # Text Colors
        'text_primary': '#2c3e50',
        'text_secondary': '#7f8c8d',
        'text_light': '#ffffff',
        'text_muted': '#95a5a6',
        
        # Gradients
        'gradient_primary': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #667eea, stop:1 #764ba2)',
        'gradient_success': 'qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #56ab2f, stop:1 #a8e6cf)',
        'gradient_error': 'qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff416c, stop:1 #ff4757)',
        'gradient_blue': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4facfe, stop:1 #00f2fe)',
    }
    
    # Light Theme Stylesheet
    LIGHT_THEME = f"""
    /* Main Window */
    QWidget {{
        background-color: {COLORS['background_main']};
        color: {COLORS['text_primary']};
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 10px;
    }}
    
    /* Main Container */
    QWidget#MainWindow {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #f8f9fa, stop:0.5 #e9ecef, stop:1 #f8f9fa);
    }}
    
    /* Menu Bar */
    QMenuBar {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                   stop:0 #ffffff, stop:1 #f1f3f4);
        border-bottom: 2px solid {COLORS['border_light']};
        padding: 4px 8px;
        font-weight: 500;
        font-size: 11px;
    }}
    
    QMenuBar::item {{
        background: transparent;
        padding: 6px 12px;
        margin: 2px;
        border-radius: 6px;
        color: {COLORS['text_primary']};
    }}
    
    QMenuBar::item:selected {{
        background: {COLORS['gradient_blue']};
        color: white;
    }}
    
    QMenuBar::item:pressed {{
        background: {COLORS['accent_blue']};
    }}
    
    QMenu {{
        background: {COLORS['background_card']};
        border: 1px solid {COLORS['border_light']};
        border-radius: 8px;
        padding: 4px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }}
    
    QMenu::item {{
        padding: 8px 16px;
        border-radius: 4px;
        margin: 1px;
    }}
    
    QMenu::item:selected {{
        background: {COLORS['gradient_blue']};
        color: white;
    }}
    
    /* Buttons - Start Button */
    QPushButton#start_button {{
        background: {COLORS['gradient_success']};
        border: none;
        border-radius: 12px;
        padding: 12px 20px;
        font-size: 14px;
        font-weight: 600;
        color: white;
        min-height: 20px;
        box-shadow: 0 4px 15px rgba(86, 171, 47, 0.3);
    }}
    
    QPushButton#start_button:hover {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4a9b2e, stop:1 #95d4a8);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(86, 171, 47, 0.4);
    }}
    
    QPushButton#start_button:pressed {{
        background: {COLORS['success']};
        transform: translateY(0px);
    }}
    
    QPushButton#start_button:disabled {{
        background: linear-gradient(135deg, #bdc3c7, #ecf0f1);
        color: {COLORS['text_muted']};
        box-shadow: none;
    }}
    
    /* Buttons - End Button */
    QPushButton#end_button {{
        background: {COLORS['gradient_error']};
        border: none;
        border-radius: 12px;
        padding: 12px 20px;
        font-size: 14px;
        font-weight: 600;
        color: white;
        min-height: 20px;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.3);
    }}
    
    QPushButton#end_button:hover {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #e63946, stop:1 #f1495a);
        box-shadow: 0 6px 20px rgba(255, 65, 108, 0.4);
    }}
    
    QPushButton#end_button:pressed {{
        background: {COLORS['error']};
    }}
    
    /* Regular Buttons */
    QPushButton {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                   stop:0 #ffffff, stop:1 #f8f9fa);
        border: 2px solid {COLORS['border_light']};
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 11px;
        font-weight: 500;
        color: {COLORS['text_primary']};
        min-height: 16px;
    }}
    
    QPushButton:hover {{
        background: {COLORS['gradient_blue']};
        color: white;
        border-color: {COLORS['accent_blue']};
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 172, 254, 0.3);
    }}
    
    QPushButton:pressed {{
        background: {COLORS['accent_blue']};
        transform: translateY(0px);
    }}
    
    QPushButton:disabled {{
        background: {COLORS['background_input']};
        color: {COLORS['text_muted']};
        border-color: {COLORS['border_light']};
    }}
    
    /* ComboBoxes */
    QComboBox {{
        background: {COLORS['background_card']};
        border: 2px solid {COLORS['border_light']};
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 11px;
        color: {COLORS['text_primary']};
        min-height: 16px;
        selection-background-color: {COLORS['accent_blue']};
    }}
    
    QComboBox:focus {{
        border-color: {COLORS['accent_blue']};
        box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1);
    }}
    
    QComboBox:hover {{
        border-color: {COLORS['accent_blue']};
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                   stop:0 #ffffff, stop:1 #f8f9fa);
    }}
    
    QComboBox::drop-down {{
        border: none;
        width: 25px;
        border-radius: 6px;
    }}
    
    QComboBox::down-arrow {{
        image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOCIgdmlld0JveD0iMCAwIDEyIDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0xIDFMNiA2TDExIDEiIHN0cm9rZT0iIzY2NzNBMCIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPC9zdmc+);
        width: 12px;
        height: 8px;
    }}
    
    QComboBox QAbstractItemView {{
        background: {COLORS['background_card']};
        border: 1px solid {COLORS['border_light']};
        border-radius: 8px;
        selection-background-color: {COLORS['gradient_blue']};
        selection-color: white;
        padding: 4px;
        outline: none;
    }}
    
    /* Line Edits */
    QLineEdit {{
        background: {COLORS['background_card']};
        border: 2px solid {COLORS['border_light']};
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 11px;
        color: {COLORS['text_primary']};
        selection-background-color: {COLORS['accent_blue']};
        min-height: 16px;
    }}
    
    QLineEdit:focus {{
        border-color: {COLORS['accent_blue']};
        box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1);
        background: {COLORS['background_card']};
    }}
    
    QLineEdit:hover {{
        border-color: {COLORS['accent_cyan']};
    }}
    
    QLineEdit:disabled {{
        background: {COLORS['background_input']};
        color: {COLORS['text_muted']};
        border-color: {COLORS['border_light']};
    }}
    
    /* Labels */
    QLabel {{
        color: {COLORS['text_primary']};
        font-size: 11px;
        font-weight: 500;
        background: transparent;
        padding: 2px;
    }}
    
    /* Media Display Label */
    QLabel#displayed_media_label {{
        background: {COLORS['background_card']};
        border: 3px dashed {COLORS['border_medium']};
        border-radius: 12px;
        color: {COLORS['text_secondary']};
        font-size: 14px;
        font-weight: 400;
        padding: 20px;
        margin: 10px;
        background-image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHZpZXdCb3g9IjAgMCA0MCA0MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTIwIDEwVjMwTTEwIDIwSDMwIiBzdHJva2U9IiNCREMzQzciIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+Cjwvc3ZnPg==);
        background-repeat: no-repeat;
        background-position: center;
        background-size: 60px 60px;
    }}
    
    /* Status Label */
    QLabel#status_label {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                   stop:0 rgba(52, 152, 219, 0.1), stop:1 rgba(155, 89, 182, 0.1));
        border: 1px solid {COLORS['info']};
        border-radius: 8px;
        padding: 12px 20px;
        font-size: 13px;
        font-weight: 600;
        color: {COLORS['info']};
        margin: 5px;
    }}
    
    /* CheckBoxes */
    QCheckBox {{
        color: {COLORS['text_primary']};
        font-size: 11px;
        font-weight: 500;
        spacing: 8px;
        padding: 4px;
    }}
    
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 2px solid {COLORS['border_medium']};
        background: {COLORS['background_card']};
    }}
    
    QCheckBox::indicator:hover {{
        border-color: {COLORS['accent_blue']};
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                   stop:0 #ffffff, stop:1 #f0f8ff);
    }}
    
    QCheckBox::indicator:checked {{
        background: {COLORS['gradient_blue']};
        border-color: {COLORS['accent_blue']};
        image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOSIgdmlld0JveD0iMCAwIDEyIDkiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0xIDQuNUw0LjUgOEwxMSAxIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4=);
    }}
    
    QCheckBox::indicator:checked:hover {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                   stop:0 #4285f4, stop:1 #34a853);
    }}
    
    /* Frames */
    QFrame {{
        border-radius: 8px;
        background: {COLORS['background_card']};
    }}
    
    /* Scroll Bars */
    QScrollBar:vertical {{
        background: {COLORS['background_input']};
        width: 12px;
        border-radius: 6px;
        margin: 0;
    }}
    
    QScrollBar::handle:vertical {{
        background: {COLORS['gradient_blue']};
        border-radius: 6px;
        min-height: 20px;
        margin: 2px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background: {COLORS['accent_blue']};
    }}
    
    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    
    /* Tooltips */
    QToolTip {{
        background: {COLORS['primary_dark']};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 10px;
        opacity: 230;
    }}
    
    /* Message Boxes */
    QMessageBox {{
        background: {COLORS['background_card']};
        color: {COLORS['text_primary']};
    }}
    
    QMessageBox QPushButton {{
        min-width: 80px;
        padding: 8px 16px;
    }}
    """
    
    # Dark Theme Stylesheet
    DARK_THEME = f"""
    /* Main Window - Dark Theme */
    QWidget {{
        background-color: {COLORS['primary_dark']};
        color: {COLORS['text_light']};
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 10px;
    }}
    
    /* Main Container */
    QWidget#MainWindow {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
    }}
    
    /* Menu Bar - Dark */
    QMenuBar {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                   stop:0 #2c3e50, stop:1 #34495e);
        border-bottom: 2px solid #4a5568;
        padding: 4px 8px;
        font-weight: 500;
        font-size: 11px;
        color: {COLORS['text_light']};
    }}
    
    QMenuBar::item {{
        background: transparent;
        padding: 6px 12px;
        margin: 2px;
        border-radius: 6px;
        color: {COLORS['text_light']};
    }}
    
    QMenuBar::item:selected {{
        background: {COLORS['gradient_blue']};
        color: white;
    }}
    
    QMenu {{
        background: {COLORS['background_dark']};
        border: 1px solid #4a5568;
        border-radius: 8px;
        padding: 4px;
        color: {COLORS['text_light']};
    }}
    
    QMenu::item {{
        padding: 8px 16px;
        border-radius: 4px;
        margin: 1px;
        color: {COLORS['text_light']};
    }}
    
    QMenu::item:selected {{
        background: {COLORS['gradient_blue']};
        color: white;
    }}
    
    /* Buttons - Start Button Dark */
    QPushButton#start_button {{
        background: {COLORS['gradient_success']};
        border: none;
        border-radius: 12px;
        padding: 12px 20px;
        font-size: 14px;
        font-weight: 600;
        color: white;
        min-height: 20px;
        box-shadow: 0 4px 15px rgba(86, 171, 47, 0.4);
    }}
    
    QPushButton#start_button:hover {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4a9b2e, stop:1 #95d4a8);
        box-shadow: 0 6px 20px rgba(86, 171, 47, 0.5);
    }}
    
    QPushButton#start_button:disabled {{
        background: linear-gradient(135deg, #4a5568, #2d3748);
        color: #718096;
        box-shadow: none;
    }}
    
    /* Buttons - End Button Dark */
    QPushButton#end_button {{
        background: {COLORS['gradient_error']};
        border: none;
        border-radius: 12px;
        padding: 12px 20px;
        font-size: 14px;
        font-weight: 600;
        color: white;
        min-height: 20px;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4);
    }}
    
    QPushButton#end_button:hover {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #e63946, stop:1 #f1495a);
        box-shadow: 0 6px 20px rgba(255, 65, 108, 0.5);
    }}
    
    /* Regular Buttons - Dark */
    QPushButton {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                   stop:0 #4a5568, stop:1 #2d3748);
        border: 2px solid #4a5568;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 11px;
        font-weight: 500;
        color: {COLORS['text_light']};
        min-height: 16px;
    }}
    
    QPushButton:hover {{
        background: {COLORS['gradient_blue']};
        color: white;
        border-color: {COLORS['accent_blue']};
        box-shadow: 0 4px 12px rgba(79, 172, 254, 0.4);
    }}
    
    QPushButton:pressed {{
        background: {COLORS['accent_blue']};
    }}
    
    QPushButton:disabled {{
        background: #2d3748;
        color: #718096;
        border-color: #4a5568;
    }}
    
    /* ComboBoxes - Dark */
    QComboBox {{
        background: {COLORS['background_dark']};
        border: 2px solid #4a5568;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 11px;
        color: {COLORS['text_light']};
        min-height: 16px;
    }}
    
    QComboBox:focus {{
        border-color: {COLORS['accent_cyan']};
        box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.2);
    }}
    
    QComboBox:hover {{
        border-color: {COLORS['accent_cyan']};
    }}
    
    QComboBox QAbstractItemView {{
        background: {COLORS['background_dark']};
        border: 1px solid #4a5568;
        border-radius: 8px;
        selection-background-color: {COLORS['accent_blue']};
        selection-color: white;
        color: {COLORS['text_light']};
    }}
    
    /* Line Edits - Dark */
    QLineEdit {{
        background: {COLORS['background_dark']};
        border: 2px solid #4a5568;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 11px;
        color: {COLORS['text_light']};
        selection-background-color: {COLORS['accent_blue']};
        min-height: 16px;
    }}
    
    QLineEdit:focus {{
        border-color: {COLORS['accent_cyan']};
        box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.2);
    }}
    
    QLineEdit:hover {{
        border-color: {COLORS['accent_cyan']};
    }}
    
    QLineEdit:disabled {{
        background: #2d3748;
        color: #718096;
        border-color: #4a5568;
    }}
    
    /* Labels - Dark */
    QLabel {{
        color: {COLORS['text_light']};
        font-size: 11px;
        font-weight: 500;
        background: transparent;
        padding: 2px;
    }}
    
    /* Media Display Label - Dark */
    QLabel#displayed_media_label {{
        background: {COLORS['background_dark']};
        border: 3px dashed #4a5568;
        border-radius: 12px;
        color: #a0aec0;
        font-size: 14px;
        font-weight: 400;
        padding: 20px;
        margin: 10px;
    }}
    
    /* Status Label - Dark */
    QLabel#status_label {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                   stop:0 rgba(0, 212, 255, 0.2), stop:1 rgba(79, 172, 254, 0.2));
        border: 1px solid {COLORS['accent_cyan']};
        border-radius: 8px;
        padding: 12px 20px;
        font-size: 13px;
        font-weight: 600;
        color: {COLORS['accent_cyan']};
        margin: 5px;
    }}
    
    /* CheckBoxes - Dark */
    QCheckBox {{
        color: {COLORS['text_light']};
        font-size: 11px;
        font-weight: 500;
        spacing: 8px;
        padding: 4px;
    }}
    
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 2px solid #4a5568;
        background: {COLORS['background_dark']};
    }}
    
    QCheckBox::indicator:hover {{
        border-color: {COLORS['accent_cyan']};
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                   stop:0 #2d3748, stop:1 #4a5568);
    }}
    
    QCheckBox::indicator:checked {{
        background: {COLORS['gradient_blue']};
        border-color: {COLORS['accent_blue']};
        image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOSIgdmlld0JveD0iMCAwIDEyIDkiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0xIDQuNUw0LjUgOEwxMSAxIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4=);
    }}
    
    /* Tooltips - Dark */
    QToolTip {{
        background: #1a1a2e;
        color: white;
        border: 1px solid {COLORS['accent_cyan']};
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 10px;
    }}
    """
    
    @staticmethod
    def get_button_style(button_type="default", size="medium"):
        """
        Generate specific button styles
        button_type: 'default', 'primary', 'success', 'warning', 'error'
        size: 'small', 'medium', 'large'
        """
        sizes = {
            'small': {'padding': '6px 12px', 'font_size': '10px', 'height': '14px'},
            'medium': {'padding': '8px 16px', 'font_size': '11px', 'height': '16px'},
            'large': {'padding': '12px 20px', 'font_size': '14px', 'height': '20px'}
        }
        
        size_style = sizes.get(size, sizes['medium'])
        
        if button_type == "primary":
            return f"""
                background: {UIStyles.COLORS['gradient_blue']};
                border: none;
                border-radius: 8px;
                padding: {size_style['padding']};
                font-size: {size_style['font_size']};
                font-weight: 600;
                color: white;
                min-height: {size_style['height']};
                box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
            """
        elif button_type == "success":
            return f"""
                background: {UIStyles.COLORS['gradient_success']};
                border: none;
                border-radius: 8px;
                padding: {size_style['padding']};
                font-size: {size_style['font_size']};
                font-weight: 600;
                color: white;
                min-height: {size_style['height']};
                box-shadow: 0 4px 15px rgba(86, 171, 47, 0.3);
            """
        elif button_type == "error":
            return f"""
                background: {UIStyles.COLORS['gradient_error']};
                border: none;
                border-radius: 8px;
                padding: {size_style['padding']};
                font-size: {size_style['font_size']};
                font-weight: 600;
                color: white;
                min-height: {size_style['height']};
                box-shadow: 0 4px 15px rgba(255, 65, 108, 0.3);
            """