# database.py
import psycopg2
from psycopg2 import sql
import datetime
import os
from dotenv import load_dotenv

# Cargar variables del archivo .env
load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        """Establecer conexión con PostgreSQL usando variables de entorno"""
        try:
            self.connection = psycopg2.connect(
                host=os.getenv('DB_HOST'),
                port=os.getenv('DB_PORT'),
                database=os.getenv('DB_NAME'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
                sslmode=os.getenv('DB_SSLMODE')
            )
            print("✅ Conexión a BD establecida")
        except Exception as e:
            print(f"❌ Error conectando a BD: {e}")
    
    def buscar_paciente(self, rut):
        """Buscar paciente por RUT"""
        try:
            cursor = self.connection.cursor()
            query = "SELECT * FROM pacientes WHERE rut = %s"
            cursor.execute(query, (rut,))
            paciente = cursor.fetchone()
            
            if paciente:
                # Convertir a diccionario
                columns = [desc[0] for desc in cursor.description]
                cursor.close()
                return dict(zip(columns, paciente))
            
            cursor.close()
            return None
        except Exception as e:
            print(f"Error buscando paciente: {e}")
            return None
    
    def registrar_paciente(self, datos_paciente):
        """Registrar nuevo paciente"""
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO pacientes 
                (rut, nombre, apellido, fecha_nacimiento, telefono, email, direccion)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (
                datos_paciente['rut'],
                datos_paciente['nombre'],
                datos_paciente['apellido'],
                datos_paciente['fecha_nacimiento'],
                datos_paciente['telefono'],
                datos_paciente['email'],
                datos_paciente['direccion']
            ))
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Error registrando paciente: {e}")
            self.connection.rollback()
            return False
    
    def obtener_imagenes_paciente(self, id_paciente):
        """Obtener todas las imágenes de un paciente"""
        try:
            cursor = self.connection.cursor()
            query = """
                SELECT i.*, c.fecha_consulta 
                FROM imagenes i
                JOIN consultas c ON i.id_consulta = c.id_consulta
                WHERE c.id_paciente = %s
                ORDER BY c.fecha_consulta DESC
            """
            cursor.execute(query, (id_paciente,))
            imagenes = cursor.fetchall()
            cursor.close()
            return imagenes
        except Exception as e:
            print(f"Error obteniendo imágenes: {e}")
            return []
    
    def guardar_analisis(self, id_paciente, datos_analisis):
        """Guardar análisis completo en la BD"""
        try:
            cursor = self.connection.cursor()
            
            # 1. Insertar consulta
            query_consulta = """
                INSERT INTO consultas 
                (id_paciente, fecha_consulta, localizacion_lesion, medico_tratante)
                VALUES (%s, %s, %s, %s)
                RETURNING id_consulta
            """
            cursor.execute(query_consulta, (
                id_paciente,
                datetime.datetime.now().date(),
                datos_analisis['localizacion'],
                datos_analisis.get('medico', 'Sistema')
            ))
            id_consulta = cursor.fetchone()[0]
            
            # 2. Insertar imagen
            query_imagen = """
                INSERT INTO imagenes 
                (id_consulta, nombre_archivo, ruta_almacenamiento, tipo_imagen)
                VALUES (%s, %s, %s, %s)
                RETURNING id_imagen
            """
            cursor.execute(query_imagen, (
                id_consulta,
                datos_analisis['nombre_archivo'],
                datos_analisis['ruta_almacenamiento'],
                datos_analisis['tipo_imagen']
            ))
            id_imagen = cursor.fetchone()[0]
            
            # 3. Insertar resultado del modelo
            query_resultado = """
                INSERT INTO resultados_modelo 
                (id_imagen, probabilidad_melanoma, diagnostico_modelo, confianza, version_modelo)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query_resultado, (
                id_imagen,
                datos_analisis['probabilidad_melanoma'],
                datos_analisis['diagnostico_modelo'],
                datos_analisis['confianza'],
                datos_analisis.get('version_modelo', 'v2.1-melanoma-detector')
            ))
            
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Error guardando análisis: {e}")
            self.connection.rollback()
            return False

    def crear_consulta(self, id_paciente, datos_consulta):
        """Crear una nueva consulta y retornar su ID"""
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO consultas 
                (id_paciente, fecha_consulta, localizacion_lesion, medico_tratante, notas_consulta)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id_consulta
            """
            cursor.execute(query, (
                id_paciente,
                datos_consulta['fecha_consulta'],
                datos_consulta['localizacion'],
                datos_consulta.get('medico', 'Sistema'),
                datos_consulta.get('notas_consulta', '')  # Notas generales de la consulta
            ))
            id_consulta = cursor.fetchone()[0]
            self.connection.commit()
            cursor.close()
            return id_consulta
        except Exception as e:
            print(f"Error creando consulta: {e}")
            self.connection.rollback()
            return None

    def agregar_imagen_a_consulta(self, id_consulta, imagen_data):
        """Agregar una imagen a una consulta existente"""
        try:
            cursor = self.connection.cursor()
            
            # Insertar imagen CON notas específicas
            query_imagen = """
                INSERT INTO imagenes 
                (id_consulta, nombre_archivo, ruta_almacenamiento, tipo_imagen, notas_imagen)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id_imagen
            """
            cursor.execute(query_imagen, (
                id_consulta,
                imagen_data['nombre_archivo'],
                imagen_data['ruta_almacenamiento'],
                imagen_data['tipo_imagen'],
                imagen_data.get('notas_imagen', '')  # Notas específicas de esta imagen
            ))
            id_imagen = cursor.fetchone()[0]
            
            # Insertar resultado del modelo
            query_resultado = """
                INSERT INTO resultados_modelo 
                (id_imagen, probabilidad_melanoma, diagnostico_modelo, confianza, version_modelo)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query_resultado, (
                id_imagen,
                imagen_data['probabilidad_melanoma'],
                imagen_data['diagnostico_modelo'],
                imagen_data['confianza'],
                imagen_data.get('version_modelo', 'v2.1-melanoma-detector')
            ))
            
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Error agregando imagen a consulta: {e}")
            self.connection.rollback()
            return False

    def obtener_ultima_consulta_paciente(self, id_paciente):
        """Obtener la última consulta de un paciente"""
        try:
            cursor = self.connection.cursor()
            query = """
                SELECT id_consulta, fecha_consulta, localizacion_lesion, notas_consulta
                FROM consultas 
                WHERE id_paciente = %s 
                ORDER BY id_consulta DESC 
                LIMIT 1
            """
            cursor.execute(query, (id_paciente,))
            consulta = cursor.fetchone()
            
            if consulta:
                columns = [desc[0] for desc in cursor.description]
                cursor.close()
                return dict(zip(columns, consulta))
            
            cursor.close()
            return None
        except Exception as e:
            print(f"Error obteniendo última consulta: {e}")
            return None

    def obtener_consultas_completas_paciente(self, id_paciente):
        """Obtener consultas completas con sus imágenes y diagnósticos - ACTUALIZADO"""
        try:
            cursor = self.connection.cursor()
            
            # Obtener consultas
            query_consultas = """
                SELECT c.id_consulta, c.fecha_consulta, c.localizacion_lesion, c.notas_consulta
                FROM consultas c
                WHERE c.id_paciente = %s
                ORDER BY c.fecha_consulta DESC
            """
            cursor.execute(query_consultas, (id_paciente,))
            consultas = cursor.fetchall()
            
            resultado = []
            for consulta in consultas:
                id_consulta = consulta[0]
                
                # --- INICIO DEL CAMBIO ---
                # Obtener imágenes y diagnósticos de esta consulta
                # AÑADIMOS: i.ruta_almacenamiento
                query_imagenes = """
                    SELECT i.nombre_archivo, i.notas_imagen, r.diagnostico_modelo, r.confianza, i.ruta_almacenamiento
                    FROM imagenes i
                    LEFT JOIN resultados_modelo r ON i.id_imagen = r.id_imagen
                    WHERE i.id_consulta = %s
                """
                # --- FIN DEL CAMBIO ---
                
                cursor.execute(query_imagenes, (id_consulta,))
                imagenes = cursor.fetchall()
                
                lista_imagenes = []
                for img in imagenes:
                    # --- INICIO DEL CAMBIO ---
                    # Añadimos 'ruta_almacenamiento' al diccionario
                    lista_imagenes.append({
                        'nombre_archivo': img[0],
                        'notas_imagen': img[1] if img[1] else "",
                        'diagnostico': img[2] if img[2] else "Sin diagnóstico",
                        'confianza': float(img[3]) if img[3] else 0.0,
                        'ruta_almacenamiento': img[4] # <-- CAMBIO CLAVE
                    })
                    # --- FIN DEL CAMBIO ---
                
                resultado.append({
                    'id_consulta': id_consulta,
                    'fecha_consulta': consulta[1],
                    'localizacion_lesion': consulta[2],
                    'notas_consulta': consulta[3],
                    'imagenes': lista_imagenes
                })
            
            cursor.close()
            return resultado
        except Exception as e:
            print(f"Error obteniendo consultas completas: {e}")
            return []

    def close(self):
        """Cerrar conexión"""
        if self.connection:
            self.connection.close()