from django.db import models

# Create your models here.
class manage_users_model(models.Model):
    User_id = models.AutoField(primary_key = True)
    user_Profile = models.FileField(upload_to = 'images/')
    User_Email = models.EmailField(max_length = 50)
    User_Status = models.CharField(max_length = 10)
    
    class Meta:
        db_table = 'manage_users'

        


class NB(models.Model):
    name = models.CharField(max_length=255)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()

    def __str__(self):
        return self.name  
    
    class Meta:
        db_table = 'NB'


class SVM(models.Model):
    name = models.CharField(max_length=255)
    accuracy = models.FloatField(max_length=100)
    precision = models.FloatField(max_length=100)
    recall = models.FloatField(max_length=100)
    f1_score = models.FloatField(max_length=100)

    def __str__(self):
        return self.name
    
    class Meta:
        db_table = 'SVM'


class LR(models.Model):
    name = models.CharField(max_length=255)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()

    def __str__(self):
        return self.name
    
    class Meta:
        db_table = 'LR'






