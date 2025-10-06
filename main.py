import multiprocessing
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from database import DatabaseManager
import datetime
import re
from tkcalendar import DateEntry

# -------------------------------
# DEFINICIONES DEL MODELO
# -------------------------------

class CNN_Exacta(nn.Module):
    def __init__(self):
        super(CNN_Exacta, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 114, kernel_size=3, padding=1),
            nn.BatchNorm2d(114),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(114, 228, kernel_size=3, padding=1),
            nn.BatchNorm2d(228),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(228, 456, kernel_size=3, padding=1),
            nn.BatchNorm2d(456),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(456, 912, kernel_size=3, padding=1),
            nn.BatchNorm2d(912),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(912 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AppSettings:
    def __init__(self):
        self.textFont = "Arial"
        self.fontSize = 12
        self.darkMode = True
        self.load_configuration()

    def load_configuration(self):
        try:
            with open("config.txt", "r") as f:
                lines = f.readlines()
                self.textFont = lines[0].split("\"")[1]
                self.fontSize = int(lines[1].split("= ")[1])
                self.darkMode = bool(int(lines[2].split("= ")[1]))
        except FileNotFoundError:
            self.save_configuration()
        except Exception as e:
            print(f"Error loading configuration: {e}")

    def save_configuration(self):
        try:
            with open("config.txt", "w") as f:
                f.write(f'letterType = "{self.textFont}"\n')
                f.write(f'letterSize = {self.fontSize}\n')
                f.write(f'darkMode = {int(self.darkMode)}\n')
        except Exception as e:
            print(f"Error saving configuration: {e}")

def load_flexible_model(model_path, device):
    """Carga el modelo de forma flexible, intentando diferentes enfoques"""
    try:
        # Agregar weights_only=True para evitar el warning
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        model = CNN_Exacta()
        
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            cleaned_key = key.replace('model.', '').replace('module.', '')
            cleaned_state_dict[cleaned_key] = value
        
        model.load_state_dict(cleaned_state_dict, strict=False)
        
        model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        print(f"Error en carga flexible: {e}")
        # También agregar weights_only=False aquí
        model = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(model, dict):
            if 'model' in model:
                model = model['model']
            elif 'state_dict' in model:
                model = CNN_Exacta()
                model.load_state_dict(model['state_dict'], strict=False)
        return model

# -------------------------------
# FUNCIONES DE VALIDACIÓN
# -------------------------------

def validar_rut(rut):
    """Validar formato de RUT (sin dígito verificador)"""
    patron = r'^\d{7,8}-[\dkK]$'
    return re.match(patron, rut) is not None

def validar_solo_letras(texto):
    """Validar que el texto contenga solo letras y espacios"""
    return texto.replace(' ', '').isalpha() if texto else False

def validar_telefono(telefono):
    """Validar formato de teléfono: +56912345678"""
    patron = r'^\+\d{11}$'
    return re.match(patron, telefono) is not None

def validar_email(email):
    """Validar formato de email"""
    if not email:  # Email es opcional
        return True
    patron = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(patron, email) is not None

def validar_direccion(direccion):
    """Validar dirección (letras, números, espacios y caracteres comunes)"""
    if not direccion:  # Dirección es opcional
        return True
    # Patrón más flexible para direcciones
    patron = r'^[a-zA-Z0-9\s.,#\-ñáéíóúüÑÁÉÍÓÚÜ]+$'
    return re.match(patron, direccion) is not None and len(direccion.strip()) >= 3

# -------------------------------
# CLASES DE LA INTERFAZ
# -------------------------------

class InicioPage(ctk.CTkFrame):
    def __init__(self, master, app_instance):
        super().__init__(master)
        self.app = app_instance
        self.db = DatabaseManager()
        
        self.configure(fg_color="transparent")
        self.pack(fill="both", expand=True, padx=50, pady=50)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Interfaz de búsqueda de RUT"""
        title = ctk.CTkLabel(
            self, 
            text="🏥 Sistema de Detección de Melanoma", 
            font=("Arial", 24, "bold")
        )
        title.pack(pady=20)
        
        rut_frame = ctk.CTkFrame(self)
        rut_frame.pack(pady=30, fill="x", padx=100)
        
        ctk.CTkLabel(
            rut_frame, 
            text="Ingrese RUT del Paciente:", 
            font=("Arial", 16)
        ).pack(pady=10)
        
        self.rut_entry = ctk.CTkEntry(
            rut_frame,
            placeholder_text="Ej: 12345678-9",
            font=("Arial", 14),
            width=200
        )
        self.rut_entry.pack(pady=10)
        self.rut_entry.bind("<Return>", lambda e: self.buscar_paciente())
        
        self.buscar_btn = ctk.CTkButton(
            rut_frame,
            text="Buscar Paciente",
            command=self.buscar_paciente,
            font=("Arial", 14)
        )
        self.buscar_btn.pack(pady=10)
        
        self.resultado_label = ctk.CTkLabel(
            rut_frame,
            text="",
            font=("Arial", 12)
        )
        self.resultado_label.pack(pady=10)
    
    def buscar_paciente(self):
        """Buscar paciente en la base de datos"""
        rut = self.rut_entry.get().strip()
        if not rut:
            messagebox.showwarning("Advertencia", "Por favor ingrese un RUT")
            return
        
        # Validar formato de RUT
        if not validar_rut(rut):
            messagebox.showerror("Error", "Formato de RUT inválido. Use: 12345678-9")
            return
        
        paciente = self.db.buscar_paciente(rut)
        
        if paciente:
            self.resultado_label.configure(
                text=f"✅ Paciente encontrado: {paciente['nombre']} {paciente['apellido']}",
                text_color="green"
            )
            self.mostrar_historial_paciente(paciente)
        else:
            self.resultado_label.configure(
                text="❌ Paciente no encontrado. Será registrado.",
                text_color="orange"
            )
            self.mostrar_registro_paciente(rut)
    
    def mostrar_historial_paciente(self, paciente):
        """Mostrar historial del paciente existente - COMPLETAMENTE CORREGIDO"""
        for widget in self.winfo_children():
            widget.destroy()
        
        title = ctk.CTkLabel(
            self, 
            text=f"📋 Historial de {paciente['nombre']} {paciente['apellido']}", 
            font=("Arial", 20, "bold")
        )
        title.pack(pady=20)
        
        # Obtener consultas del paciente con sus imágenes
        consultas_completas = self.db.obtener_consultas_completas_paciente(paciente['id_paciente'])
        
        if consultas_completas:
            ctk.CTkLabel(
                self,
                text=f"📊 {len(consultas_completas)} consulta(s) encontrada(s)",
                font=("Arial", 14)
            ).pack(pady=10)
            
            for consulta in consultas_completas:
                consulta_frame = ctk.CTkFrame(self, fg_color="#2b2b2b")
                consulta_frame.pack(fill="x", padx=50, pady=8)
                
                # Información de la consulta
                info_text = f"📅 Consulta del {consulta['fecha_consulta']} - Localización: {consulta['localizacion_lesion']}"
                if consulta['notas_consulta']:
                    info_text += f" - 📝 Notas: {consulta['notas_consulta']}"
                
                ctk.CTkLabel(
                    consulta_frame,
                    text=info_text,
                    font=("Arial", 12, "bold"),
                    wraplength=800
                ).pack(anchor="w", padx=10, pady=5)
                
                # Mostrar imágenes de esta consulta
                if consulta['imagenes']:
                    for img in consulta['imagenes']:
                        img_frame = ctk.CTkFrame(consulta_frame, fg_color="#3b3b3b")
                        img_frame.pack(fill="x", padx=20, pady=2)
                        
                        img_text = f"  📷 {img['nombre_archivo']} - Diagnóstico: {img['diagnostico']} (Confianza: {img['confianza']:.2%})"
                        ctk.CTkLabel(
                            img_frame,
                            text=img_text,
                            font=("Arial", 10),
                            wraplength=750
                        ).pack(anchor="w", padx=10, pady=2)
                else:
                    ctk.CTkLabel(
                        consulta_frame,
                        text="  No hay imágenes en esta consulta",
                        font=("Arial", 10),
                        text_color="gray"
                    ).pack(anchor="w", padx=20, pady=2)
        else:
            ctk.CTkLabel(
                self,
                text="No hay consultas previas registradas",
                font=("Arial", 12),
                text_color="gray"
            ).pack(pady=20)
        
        ctk.CTkButton(
            self,
            text="🩺 Realizar Nuevo Diagnóstico",
            command=lambda: self.ir_a_analisis(paciente),
            font=("Arial", 16),
            height=40
        ).pack(pady=30)
    
    def mostrar_registro_paciente(self, rut):
        """Mostrar formulario de registro para nuevo paciente"""
        for widget in self.winfo_children():
            widget.destroy()
        
        title = ctk.CTkLabel(
            self, 
            text="📝 Registrar Nuevo Paciente", 
            font=("Arial", 20, "bold")
        )
        title.pack(pady=20)
        
        form_frame = ctk.CTkScrollableFrame(self, height=500)
        form_frame.pack(pady=20, padx=100, fill="both", expand=True)
        
        campos = [
            ("RUT", "rut", rut, True),
            ("Nombre*", "nombre", "", False),
            ("Apellido*", "apellido", "", False),
            ("Fecha Nacimiento*", "fecha_nacimiento", "", False),
            ("Teléfono", "telefono", "", False),
            ("Email", "email", "", False),
            ("Dirección", "direccion", "", False)
        ]
        
        self.entries = {}
        self.placeholder_activo = {"telefono": True}  # Control para placeholder de teléfono
        
        for label, key, default, disabled in campos:
            # Frame para cada campo
            field_frame = ctk.CTkFrame(form_frame)
            field_frame.pack(fill="x", pady=8)
            
            ctk.CTkLabel(field_frame, text=label, font=("Arial", 12)).pack(anchor="w", pady=2)
            
            if key == "fecha_nacimiento":
                # Usar DateEntry para fecha de nacimiento
                entry = DateEntry(
                    field_frame,
                    width=30,
                    background='darkblue',
                    foreground='white',
                    borderwidth=2,
                    date_pattern='yyyy-mm-dd',
                    font=("Arial", 12)
                )
                entry.pack(fill="x", pady=2)
                self.entries[key] = entry
            else:
                entry = ctk.CTkEntry(
                    field_frame,
                    placeholder_text="+56912345678" if key == "telefono" else label,
                    font=("Arial", 12),
                    height=35
                )
                entry.pack(fill="x", pady=2)
                
                # Configurar placeholder para teléfono
                if key == "telefono":
                    entry.bind("<FocusIn>", lambda e, k=key: self.borrar_placeholder_telefono(k))
                    entry.bind("<FocusOut>", lambda e, k=key: self.verificar_placeholder_telefono(k))
                
                # Validación en tiempo real para algunos campos
                if key == "telefono":
                    entry.bind("<KeyRelease>", lambda e, k=key: self.validar_campo_tiempo_real(k))
                elif key in ["nombre", "apellido", "email", "direccion"]:
                    entry.bind("<KeyRelease>", lambda e, k=key: self.validar_campo_tiempo_real(k))
                
                if default and key != "fecha_nacimiento":
                    entry.insert(0, default)
                if disabled:
                    entry.configure(state="disabled")
                
                self.entries[key] = entry
        
        # Nota sobre campos obligatorios
        ctk.CTkLabel(
            form_frame, 
            text="* Campos obligatorios", 
            font=("Arial", 10),
            text_color="orange"
        ).pack(anchor="w", pady=10)
        
        btn_frame = ctk.CTkFrame(form_frame)
        btn_frame.pack(pady=20)
        
        ctk.CTkButton(
            btn_frame,
            text="💾 Registrar Paciente",
            command=self.registrar_paciente,
            font=("Arial", 14)
        ).pack(side="left", padx=10)
        
        self.registro_status = ctk.CTkLabel(form_frame, text="", font=("Arial", 12))
        self.registro_status.pack(pady=10)
    
    def borrar_placeholder_telefono(self, campo):
        """Borrar el placeholder del teléfono cuando se hace clic"""
        if self.placeholder_activo.get(campo, False):
            self.entries[campo].delete(0, 'end')
            self.placeholder_activo[campo] = False
    
    def verificar_placeholder_telefono(self, campo):
        """Restaurar placeholder si el campo está vacío"""
        if campo == "telefono" and not self.entries[campo].get():
            self.entries[campo].configure(placeholder_text="+56912345678")
            self.placeholder_activo[campo] = True
    
    def validar_campo_tiempo_real(self, campo):
        """Validación en tiempo real para campos con colores"""
        if campo == "fecha_nacimiento":
            return
        
        entry = self.entries[campo]
        valor = entry.get().strip()
        
        # Si es teléfono y está vacío, no validar (es opcional)
        if campo == "telefono" and not valor:
            entry.configure(border_color="#979DA2")  # Color por defecto
            return
        
        # Validar según el campo
        if campo == "nombre":
            valido = validar_solo_letras(valor) if valor else False
        elif campo == "apellido":
            valido = validar_solo_letras(valor) if valor else False
        elif campo == "telefono":
            valido = validar_telefono(valor) if valor else True
        elif campo == "email":
            valido = validar_email(valor)
        elif campo == "direccion":
            valido = validar_direccion(valor)
        else:
            valido = True
        
        # Cambiar color del borde
        if campo in ["nombre", "apellido"] and not valor:
            # Campos obligatorios vacíos
            entry.configure(border_color="red")
        elif valido:
            entry.configure(border_color="green")
        else:
            entry.configure(border_color="red")
    
    def validar_campos_completos(self):
        """Validar que todos los campos estén completos y correctos"""
        datos = {}
        errores = []
        
        for key, entry in self.entries.items():
            if key == "fecha_nacimiento":
                value = entry.get_date().strftime('%Y-%m-%d')
            else:
                value = entry.get().strip()
            
            datos[key] = value
            
            # Validaciones específicas
            if key in ["nombre", "apellido"] and not value:
                errores.append(f"El campo {key} es obligatorio")
            elif key == "nombre" and value and not validar_solo_letras(value):
                errores.append("El nombre solo puede contener letras")
            elif key == "apellido" and value and not validar_solo_letras(value):
                errores.append("El apellido solo puede contener letras")
            elif key == "telefono" and value and not validar_telefono(value):
                errores.append("Teléfono debe tener formato: +56912345678")
            elif key == "email" and value and not validar_email(value):
                errores.append("Formato de email inválido")
            elif key == "direccion" and value and not validar_direccion(value):
                errores.append("La dirección debe tener al menos 3 caracteres válidos")
        
        return datos, errores
    
    def registrar_paciente(self):
        """Registrar nuevo paciente en la BD"""
        try:
            datos, errores = self.validar_campos_completos()
            
            if errores:
                messagebox.showerror("Error de validación", "\n".join(errores))
                return
            
            if self.db.registrar_paciente(datos):
                self.registro_status.configure(
                    text="✅ Paciente registrado exitosamente",
                    text_color="green"
                )
                # Buscar el paciente recién registrado y redirigir automáticamente
                messagebox.showinfo("Éxito", "Paciente registrado correctamente. Redirigiendo al diagnóstico...")
                paciente = self.db.buscar_paciente(datos['rut'])
                if paciente:
                    self.ir_a_analisis(paciente)
            else:
                messagebox.showerror("Error", "No se pudo registrar el paciente en la base de datos")
                self.registro_status.configure(
                    text="❌ Error al registrar paciente",
                    text_color="red"
                )
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al registrar: {str(e)}")
    
    def ir_a_analisis(self, paciente):
        """Ir a la pantalla de análisis con datos del paciente"""
        self.app.mostrar_analisis_page(paciente)

class AnalisisFotoPage(ctk.CTkFrame):
    def __init__(self, master, paciente, db, app_instance):  # CORREGIDO: agregar app_instance
        super().__init__(master)
        
        self.paciente = paciente
        self.db = db
        self.app = app_instance  # CORREGIDO: guardar referencia a app
        self.app_settings = AppSettings()
        self.textFont = self.app_settings.textFont
        self.fontSize = self.app_settings.fontSize
        
        self.model = None
        self.model_path = None
        self.image_path = None
        self.current_image = None
        self.id_consulta_actual = None  # Para manejar una sola consulta
        self.consulta_iniciada = False  # Controlar si ya se creó una consulta
        self.texto_placeholder_activo = True
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        self.configure(fg_color="transparent")
        self.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.setup_ui()
        self.try_auto_load_model()
    
    def setup_ui(self):
        """Configurar interfaz de usuario con nueva lógica de dos botones"""
        # Información del paciente
        info_paciente = ctk.CTkFrame(self)
        info_paciente.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            info_paciente,
            text=f"Paciente: {self.paciente['nombre']} {self.paciente['apellido']} - RUT: {self.paciente['rut']}",
            font=(self.textFont, 14, "bold")
        ).pack(pady=5)
        
        # Información de la consulta actual
        self.info_consulta_frame = ctk.CTkFrame(self)
        self.info_consulta_frame.pack(fill="x", pady=5)
        
        self.info_consulta_label = ctk.CTkLabel(
            self.info_consulta_frame,
            text="🆕 Consulta no iniciada - Complete la información de la consulta",
            font=(self.textFont, 12),
            text_color="orange"
        )
        self.info_consulta_label.pack(pady=5)
        
        # Título
        title = ctk.CTkLabel(
            self, 
            text="🔬 Detector de Melanoma", 
            font=(self.textFont, 24, "bold")
        )
        title.pack(pady=10)
        
        # Contenedor principal
        main_container = ctk.CTkScrollableFrame(self, height=600)
        main_container.pack(fill="both", expand=True, pady=10)
        
        # ==================== PASO 1: CARGAR MODELO ====================
        step1_frame = ctk.CTkFrame(main_container)
        step1_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(step1_frame, text="1. Cargar Modelo Entrenado", 
                   font=(self.textFont, 16, "bold")).pack(anchor="w", padx=10, pady=5)
        
        self.load_btn = ctk.CTkButton(
            step1_frame,
            text="Seleccionar Modelo",
            command=self.cargar_modelo,
            font=(self.textFont, self.fontSize)
        )
        self.load_btn.pack(pady=10, padx=10)
        
        self.model_status = ctk.CTkLabel(
            step1_frame, 
            text="⏳ Esperando modelo...", 
            text_color="orange",
            font=(self.textFont, 12)
        )
        self.model_status.pack(pady=5)
        
        # ==================== PASO 2: ANALIZAR IMAGEN ====================
        step2_frame = ctk.CTkFrame(main_container)
        step2_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(step2_frame, text="2. Analizar Imagen", 
                   font=(self.textFont, 16, "bold")).pack(anchor="w", padx=10, pady=5)
        
        self.image_btn = ctk.CTkButton(
            step2_frame,
            text="Seleccionar Imagen",
            command=self.seleccionar_imagen,
            state="disabled",
            font=(self.textFont, self.fontSize)
        )
        self.image_btn.pack(pady=10, padx=10)
        
        self.preview_frame = ctk.CTkFrame(step2_frame, height=200)
        self.preview_frame.pack(pady=10, padx=10, fill="x")
        self.preview_frame.pack_propagate(False)
        
        self.preview_label = ctk.CTkLabel(
            self.preview_frame, 
            text="Imagen no seleccionada",
            text_color="gray",
            font=(self.textFont, 10)
        )
        self.preview_label.pack(pady=20)
        
        # ==================== PASO 3: RESULTADOS ====================
        step3_frame = ctk.CTkFrame(main_container)
        step3_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(step3_frame, text="3. Resultados", 
                font=(self.textFont, 16, "bold")).pack(anchor="w", padx=10, pady=5)

        self.results_frame = ctk.CTkFrame(step3_frame)
        self.results_frame.pack(pady=10, padx=10, fill="x")
        
        self.result_label = ctk.CTkLabel(
            self.results_frame, 
            text="Seleccione y analice una imagen para ver resultados",
            font=(self.textFont, 14),
            text_color="gray"
        )
        self.result_label.pack(pady=10)
        
        # ==================== PASO 4: INFORMACIÓN DE LA CONSULTA ====================
        step4_frame = ctk.CTkFrame(main_container)
        step4_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(step4_frame, text="4. Información de la Consulta", 
                   font=(self.textFont, 16, "bold")).pack(anchor="w", padx=10, pady=5)
        
        # Localización de la lesión
        ctk.CTkLabel(step4_frame, text="Localización de la lesión:", 
                   font=(self.textFont, 12)).pack(anchor="w", padx=10)
        
        self.localizacion_combobox = ctk.CTkComboBox(
            step4_frame,
            values=[
                "Rostro", "Cuello", "Espalda", "Pecho", "Abdomen",
                "Brazo derecho", "Brazo izquierdo", "Pierna derecha", 
                "Pierna izquierda", "Mano", "Pie", "Cuero cabelludo"
            ],
            font=(self.textFont, 12),
            width=200,
            state="readonly"
        )
        self.localizacion_combobox.pack(pady=5, padx=10)
        self.localizacion_combobox.set("Seleccione localización")
        self.localizacion_combobox.bind("<<ComboboxSelected>>", self.actualizar_botones)
        
        # Notas de la consulta
        ctk.CTkLabel(step4_frame, text="Notas/Observaciones de la consulta:", 
                   font=(self.textFont, 12)).pack(anchor="w", padx=10, pady=(10,0))
        
        self.notas_text = ctk.CTkTextbox(
            step4_frame,
            height=80,
            font=(self.textFont, 11)
        )
        self.notas_text.pack(pady=5, padx=10, fill="x")
        self.notas_text.insert("1.0", "Ingrese observaciones adicionales aquí...")
        self.notas_text.bind("<FocusIn>", self.borrar_placeholder)
        self.notas_text.bind("<KeyRelease>", self.actualizar_botones)
        
        # BOTONES DE ACCIÓN - NUEVA LÓGICA
        botones_frame = ctk.CTkFrame(step4_frame)
        botones_frame.pack(pady=10, padx=10, fill="x")
        
        # Botón para guardar imagen actual
        self.guardar_imagen_btn = ctk.CTkButton(
            botones_frame,
            text="💾 Guardar Imagen en Consulta Actual",
            command=self.guardar_imagen_actual,
            state="disabled",
            font=(self.textFont, self.fontSize),
            fg_color="#1f538d"
        )
        self.guardar_imagen_btn.pack(pady=5, fill="x")
        
        # Botón para finalizar consulta
        self.finalizar_consulta_btn = ctk.CTkButton(
            botones_frame,
            text="✅ Finalizar Consulta",
            command=self.finalizar_consulta,
            state="disabled",
            font=(self.textFont, self.fontSize),
            fg_color="green"
        )
        self.finalizar_consulta_btn.pack(pady=5, fill="x")
        
        # Información del modelo
        info_frame = ctk.CTkFrame(main_container)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(info_frame, text="ℹ️ Información del Modelo", 
                   font=(self.textFont, 14, "bold")).pack(anchor="w", padx=10, pady=5)
        
        self.info_text = ctk.CTkTextbox(info_frame, height=80, font=("Consolas", 10))
        self.info_text.pack(fill="x", padx=10, pady=5)
        self.info_text.insert("1.0", "Cargue un modelo para ver la información...")
        self.info_text.configure(state="disabled")
        
        # Botones de navegación
        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=10)
        
        ctk.CTkButton(
            btn_frame, 
            text="🔙 Volver al Inicio", 
            command=self.volver_inicio,
            font=(self.textFont, self.fontSize)
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            btn_frame, 
            text="Salir", 
            command=self.quit_app, 
            fg_color="red",
            font=(self.textFont, self.fontSize)
        ).pack(side="left", padx=10)

    def borrar_placeholder(self, event):
        """Borrar el texto placeholder cuando el usuario hace clic"""
        if self.texto_placeholder_activo:
            self.notas_text.delete("1.0", "end")
            self.texto_placeholder_activo = False

    def actualizar_botones(self, event=None):
        """Actualizar estado de los botones según las condiciones"""
        tiene_modelo = self.model is not None
        localizacion_seleccionada = self.localizacion_combobox.get() != "Seleccione localización"
        tiene_imagen_analizada = self.image_path is not None
        
        # Botón de guardar imagen: necesita modelo, localización e imagen analizada
        if tiene_modelo and localizacion_seleccionada and tiene_imagen_analizada:
            self.guardar_imagen_btn.configure(state="normal")
        else:
            self.guardar_imagen_btn.configure(state="disabled")
        
        # Botón de finalizar consulta: necesita que la consulta esté iniciada
        if self.consulta_iniciada:
            self.finalizar_consulta_btn.configure(state="normal")
        else:
            self.finalizar_consulta_btn.configure(state="disabled")

    def crear_consulta(self):
        """Crear una nueva consulta en la base de datos"""
        try:
            localizacion = self.localizacion_combobox.get()
            notas = self.notas_text.get("1.0", "end-1c").strip()
            if notas == "Ingrese observaciones adicionales aquí...":
                notas = ""
            
            datos_consulta = {
                'fecha_consulta': datetime.datetime.now().date(),
                'localizacion': localizacion,
                'notas_consulta': notas
            }
            
            self.id_consulta_actual = self.db.crear_consulta(self.paciente['id_paciente'], datos_consulta)
            
            if self.id_consulta_actual:
                self.consulta_iniciada = True
                self.info_consulta_label.configure(
                    text=f"✅ Consulta #{self.id_consulta_actual} iniciada - Puede agregar imágenes",
                    text_color="green"
                )
                self.actualizar_botones()
                return True
            else:
                messagebox.showerror("Error", "No se pudo crear la consulta en la base de datos")
                return False
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al crear consulta: {str(e)}")
            return False

    def guardar_imagen_actual(self):
        """Guardar la imagen actual en la consulta"""
        if not self.consulta_iniciada:
            # Si no hay consulta iniciada, crear una nueva
            if not self.crear_consulta():
                return
        
        try:
            # Preparar datos de la imagen
            imagen_data = {
                'nombre_archivo': os.path.basename(self.image_path),
                'ruta_almacenamiento': os.path.dirname(self.image_path),
                'tipo_imagen': os.path.splitext(self.image_path)[1][1:],
                'probabilidad_melanoma': getattr(self, 'probabilidad_melanoma', 0.0),
                'diagnostico_modelo': getattr(self, 'diagnostico_actual', 'Sin diagnóstico'),
                'confianza': getattr(self, 'confianza_actual', 0.0)
            }
            
            if self.db.agregar_imagen_a_consulta(self.id_consulta_actual, imagen_data):
                messagebox.showinfo("Éxito", f"Imagen guardada en consulta #{self.id_consulta_actual}")
                
                # Limpiar campos para nueva imagen
                self.limpiar_campos_imagen()
                
            else:
                messagebox.showerror("Error", "No se pudo guardar la imagen en la base de datos")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar imagen: {str(e)}")

    def limpiar_campos_imagen(self):
        """Limpiar campos relacionados con la imagen actual"""
        self.image_path = None
        self.current_image = None
        self.preview_label.configure(image=None, text="Imagen no seleccionada")
        self.result_label.configure(
            text="Seleccione y analice una imagen para ver resultados",
            text_color="gray"
        )
        self.actualizar_botones()

    def finalizar_consulta(self):
        """Finalizar la consulta actual y volver al inicio"""
        if messagebox.askyesno("Finalizar Consulta", 
                             "¿Está seguro de que desea finalizar la consulta?\n\nSe perderán los datos no guardados."):
            
            # Mostrar resumen de la consulta
            consulta_info = self.db.obtener_ultima_consulta_paciente(self.paciente['id_paciente'])
            if consulta_info and consulta_info['id_consulta'] == self.id_consulta_actual:
                messagebox.showinfo(
                    "Consulta Finalizada", 
                    f"Consulta #{self.id_consulta_actual} finalizada correctamente.\n"
                    f"Localización: {consulta_info['localizacion_lesion']}\n"
                    f"Fecha: {consulta_info['fecha_consulta']}"
                )
            
            # Volver al inicio
            self.volver_inicio()

    def try_auto_load_model(self):
        """Intenta cargar el modelo automáticamente si existe"""
        possible_model_names = [
            "melanoma_model.pth",
            "model.pth",
            "cnn_model.pth",
            "skin_cancer_model.pth",
            "modelo.pth"
        ]
        
        for model_name in possible_model_names:
            if os.path.exists(model_name):
                try:
                    self.cargar_modelo_desde_ruta(model_name)
                    return True
                except Exception as e:
                    print(f"Error al cargar {model_name}: {e}")
        return False

    def cargar_modelo_desde_ruta(self, model_path):
        """Carga un modelo desde una ruta específica"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            model = load_flexible_model(model_path, device)
            
            self.model = model
            self.model_path = model_path
            self.model_status.configure(
                text=f"✅ Modelo cargado: {os.path.basename(model_path)}", 
                text_color="green"
            )
            self.image_btn.configure(state="normal")
            self.actualizar_info()
            
        except Exception as e:
            raise Exception(f"Error al cargar el modelo: {str(e)}")

    def cargar_modelo(self):
        """Permite al usuario seleccionar el archivo del modelo"""
        model_path = filedialog.askopenfilename(
            title="Seleccionar modelo entrenado",
            filetypes=[
                ("Modelos PyTorch", "*.pth"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if model_path:
            try:
                self.cargar_modelo_desde_ruta(model_path)
                messagebox.showinfo("Éxito", "Modelo cargado correctamente")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el modelo: {str(e)}")
                self.model_status.configure(text="❌ Error al cargar el modelo", text_color="red")

    def seleccionar_imagen(self):
        """Permite al usuario seleccionar una imagen para analizar"""
        if not self.model:
            messagebox.showwarning("Advertencia", "Primero debe cargar un modelo")
            return
            
        path_image = filedialog.askopenfilename(
            title="Seleccionar imagen para analizar",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if path_image:
            self.image_path = path_image
            self.mostrar_vista_previa(path_image)
            self.analizar_imagen()

    def mostrar_vista_previa(self, image_path):
        """Muestra una vista previa de la imagen seleccionada"""
        try:
            img = Image.open(image_path)
            self.current_image = img.copy()
            img.thumbnail((180, 180))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            self.preview_label.configure(image=ctk_img, text="")
        except Exception as e:
            self.preview_label.configure(text="❌ Error en vista previa", text_color="red")

    def analizar_imagen(self):
        """Analiza la imagen seleccionada con el modelo cargado"""
        if not self.model or not self.image_path:
            messagebox.showwarning("Advertencia", "Debe cargar el modelo y seleccionar una imagen primero")
            return
            
        try:
            if self.current_image:
                image = self.current_image.convert('RGB')
            else:
                image = Image.open(self.image_path).convert('RGB')
                
            input_tensor = self.transform(image).unsqueeze(0)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
            
            img_display = image.copy()
            img_display.thumbnail((300, 300))
            ctk_img = ctk.CTkImage(light_image=img_display, dark_image=img_display, size=img_display.size)
            
            self.preview_label.configure(image=ctk_img)
            self.preview_label.image = ctk_img
            
            if predicted_class == 0:
                result_text = "MELANOMA DETECTADO"
                confidence = probabilities[0] * 100
                color = "red"
                self.probabilidad_melanoma = float(probabilities[0])
                self.diagnostico_actual = "MELANOMA"
            else:
                result_text = "NO-MELANOMA"
                confidence = probabilities[1] * 100
                color = "green"
                self.probabilidad_melanoma = float(probabilities[0])
                self.diagnostico_actual = "NO-MELANOMA"
            
            self.confianza_actual = float(max(probabilities))
            
            self.result_label.configure(
                text=f"{result_text}\nProbabilidad: {confidence:.2f}%",
                text_color=color,
                font=(self.textFont, 16, "bold")
            )
            
            # Actualizar estado de los botones
            self.actualizar_botones()
            
            messagebox.showinfo("Análisis completado", 
                              f"Resultado: {result_text}\nConfianza: {confidence:.2f}%\n\nAhora puede guardar la imagen en la consulta.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error durante el análisis: {str(e)}")

    def actualizar_info(self):
        """Actualizar información del modelo en la interfaz"""
        if self.model is None:
            info = "Ningún modelo cargado"
        else:
            info = f"Framework: PyTorch\n"
            info += f"Versión: {torch.__version__}\n"
            info += f"Tarea: Clasificación binaria\n"
            info += f"Arquitectura: CNN_Exacta (114-228-456-912)\n"
            info += f"Archivo: {os.path.basename(self.model_path) if self.model_path else 'N/A'}\n"
            info += f"Clases: 0=Melanoma, 1=No-melanoma"
        
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", info)
        self.info_text.configure(state="disabled")

    def volver_inicio(self):
        """Volver a la pantalla de inicio - CORREGIDO"""
        if hasattr(self, 'app') and self.app:
            self.app.mostrar_inicio_page()
        else:
            # Fallback: usar el master (ventana principal)
            self.master.mostrar_inicio_page()

    def quit_app(self):
        """Salir de la aplicación"""
        if messagebox.askyesno("Salir", "¿Está seguro de que quiere salir?"):
            self.master.destroy()

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("🔬 Detector de Melanoma")
        self.geometry("1000x800")
        self.resizable(True, True)
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        self.db = DatabaseManager()
        self.mostrar_inicio_page()
    
    def mostrar_inicio_page(self):
        """Mostrar página de inicio"""
        self.limpiar_pantalla()
        self.inicio_page = InicioPage(self, self)
    
    def mostrar_analisis_page(self, paciente):
        """Mostrar página de análisis - CORREGIDO: pasar self como app_instance"""
        self.limpiar_pantalla()
        self.analisis_page = AnalisisFotoPage(self, paciente, self.db, self)  # CORREGIDO: agregar self
    
    def limpiar_pantalla(self):
        """Limpiar todos los widgets"""
        for widget in self.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = App()
    app.mainloop()
