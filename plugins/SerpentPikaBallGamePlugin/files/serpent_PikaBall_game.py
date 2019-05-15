from serpent.game import Game

from .api.api import PikaBallAPI

from serpent.utilities import Singleton




class SerpentPikaBallGame(Game, metaclass=Singleton):

    def __init__(self, **kwargs):
        kwargs["platform"] = "executable"
        kwargs["window_name"] = "Pikachu VolleyBall-Serpentai"
        kwargs["executable_path"] = "C:/Python/SerpentAI/pikaball3.exe"
        
        super().__init__(**kwargs)

        self.api_class = PikaBallAPI
        self.api_instance = None

        #self.frame_transformation_pipeline_string = "RESIZE:162x114|GRAYSCALE|FLOAT"
        #self.frame_height = 114
        #self.frame_width = 162
        self.frame_transformation_pipeline_string = "RESIZE:100x100|GRAYSCALE|FLOAT"
        self.frame_height = 100
        self.frame_width = 100

    @property
    def screen_regions(self):
        regions = {
            "SAMPLE_REGION": (0, 0, 0, 0)
        }

        return regions

    @property
    def ocr_presets(self):
        presets = {
            "SAMPLE_PRESET": {
                "extract": {
                    "gradient_size": 1,
                    "closing_size": 1
                },
                "perform": {
                    "scale": 10,
                    "order": 1,
                    "horizontal_closing": 1,
                    "vertical_closing": 1
                }
            }
        }

        return presets
