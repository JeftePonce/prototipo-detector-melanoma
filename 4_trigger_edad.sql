-- 2. Crea el trigger que ejecuta la función

CREATE TRIGGER trigger_actualizar_edad
    BEFORE INSERT OR UPDATE ON pacientes
    FOR EACH ROW
    EXECUTE FUNCTION actualizar_edad();