from camera_web.db.models import Face, Person


# Select all Person objects
persons = Person.objects.all()
print(persons)
