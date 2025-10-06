-- 1. Crea función edad
CREATE OR REPLACE FUNCTION actualizar_edad()
RETURNS TRIGGER AS $$
BEGIN
    -- Calcula la edad en base a la fecha de nacimiento
    NEW.edad = EXTRACT(YEAR FROM AGE(NEW.fecha_nacimiento));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;