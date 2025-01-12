import logging
import os

# Funzione per creare il logger
def get_logger(name='atcns_logger', log_file='atcnsProject.log', level=logging.DEBUG):
    """
    Crea un logger che scrive sia su console che su file.
    
    :param name: Nome del logger
    :param log_file: Nome del file di log
    :param level: Livello di log (default DEBUG)
    :return: Logger configurato
    """
    # Crea la cartella "logs" se non esiste
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_file)

    # Crea un logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Formato del log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Handler per la console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Handler per il file
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Aggiungi gli handler al logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
