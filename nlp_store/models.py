from django.db import models

class Product(models.Model):
    product = models.CharField(max_length=100)
    description = models.TextField()
    category = models.CharField(max_length=100)
    image_path = models.ImageField(upload_to='product_images/')

    class Meta:
        app_label = 'nlp_store'

    def __str__(self):
        return self.name
