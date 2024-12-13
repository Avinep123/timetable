from django.urls import path
from .views import *
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    # Home page and timetable generation
    path('', home, name='home'),
    path('timetableGeneration/', timetable, name='timetable'),

    # Instructor management
    path('instructorAdd/', instructorAdd, name='instructorAdd'),
    path('instructorEdit/', instructorEdit, name='instructorEdit'),
    path('instructorDelete/<int:pk>/', instructorDelete, name='deleteinstructor'),

    # Room management
    path('roomAdd/', roomAdd, name='roomAdd'),
    path('roomEdit/', roomEdit, name='roomEdit'),
    path('roomDelete/<int:pk>/', roomDelete, name='deleteroom'),

    # Meeting time management
    path('meetingTimeAdd/', meetingTimeAdd, name='meetingTimeAdd'),
    path('meetingTimeEdit/', meetingTimeEdit, name='meetingTimeEdit'),
    path('meetingTimeDelete/<int:pk>/', meetingTimeDelete, name='deletemeetingtime'),

    # Course management
    path('courseAdd/', courseAdd, name='courseAdd'),
    path('courseEdit/', courseEdit, name='courseEdit'),
    path('courseDelete/<int:pk>/', courseDelete, name='deletecourse'),

    # Department management
    path('departmentAdd/', departmentAdd, name='departmentAdd'),
    path('departmentEdit/', departmentEdit, name='departmentEdit'),
    path('departmentDelete/<int:pk>/', departmentDelete, name='deletedepartment'),

    # Section management
    path('sectionAdd/', sectionAdd, name='sectionAdd'),
    path('sectionEdit/', sectionEdit, name='sectionEdit'),
    path('sectionDelete/<int:pk>/', sectionDelete, name='deletesection'),

    # API endpoints for timetable generation
    path('api/genNum/', apiGenNum, name='apiGenNum'),
    path('api/terminateGens/', apiterminateGens, name='apiterminateGens')
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
