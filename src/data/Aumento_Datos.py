import albumentations as A
from dotenv import load_dotenv



class Aumento_Datos:
    """
    Aumento de datos: transformacións elásticas, de brillo e afíns.
    
    Args:
        epocas_totais: numero total de epocas do adestramento
        epocas_quecemento: numero de epocas ate chegar á probabilidade cruceiro
        probabilidade_inicial: a probabilidade de aumento de datos na primeira epoca
        probabilidade_final: a proababilidade de aumento de datos una vez rematado o quecemento
    """
    
    def __init__(
        self,
        epocas_totais: int,
        epocas_quecemento: int = None,
        probabilidade_inicial: float = 0.1,
        probabilidade_final: float = 0.9
    ):
        self.epocas_totais = epocas_totais
        self.epocas_quecemento = epocas_quecemento if epocas_quecemento is not None else epocas_totais // 2
        self.probabilidade_inicial = probabilidade_inicial
        self.probabilidade_final = probabilidade_final
        
        if not 0 <= probabilidade_inicial <= 1:
            raise ValueError(f"A probabilidade inicial debe estar en [0, 1], dechesme {self.probabilidade_inicial}")
        if not 0 <= probabilidade_inicial <= 1:
            raise ValueError(f"A probabilidade final debe estar en [0, 1], dechesme {self.probabilidade_final}")
        if probabilidade_final > probabilidade_final:
            raise ValueError(f"A probabilidade inicial ({self.probabilidade_inicial}) debe ser menor ou igual que a final ({self.probabilidade_final}). E non é o caso.")
        if self.epocas_quecemento > epocas_totais:
            raise ValueError(f"As épocas de quecemento ({self.epocas_quecemento}) deben iguais ou menores que as totais ({self.epocas_totais})")

        load_dotenv()
    
    def get_probabilidade(self, epoca: int) -> float:
        # probabilidade cruceiro
        if epoca >= self.epocas_quecemento:
            return self.probabilidade_inicial
        
        # interpolacion lineal durante o quecemento
        progreso = epoca / self.epocas_quecemento
        return self.probabilidade_inicial + (self.probabilidade_final - self.probabilidade_inicial) * progreso
    
    def create_augmentation_pipeline(self, epoca: int) -> A.Compose:
        """
        Create augmentation pipeline with epoca-dependent probabilities.
        
        Args:
            epoca: Current epoch number (0-indexed)
            
        Returns:
            Albumentations Compose object with adjusted probabilities
        """
        p = self.get_probabilidade(epoca)
        
        return A.Compose([
            A.ElasticTransform(
                alpha=5, 
                sigma=50, 
                p=p
            ),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1),
                rotate=(-15, 15),
                shear=(-5, 5),
                p=p
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=p
            ),
        ])
    
    def __repr__(self):
        return (
            f"AugmentationScheduler("
            f"epocas_totais={self.epocas_totais}, "
            f"warmup_epocas={self.warmup_epocas}, "
            f"start_p={self.start_p}, "
            f"end_p={self.end_p})"
        )
