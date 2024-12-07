from django.http.response import HttpResponse
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import *
from .forms import *
from collections import defaultdict
import random
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
from django.conf import settings

POPULATION_SIZE = 300
C1 = 1.5  # Cognitive component
C2 = 1.5  # Social component

# Number of particles for PSO
NUM_PARTICLES = 100 
NUMB_OF_ELITE_SCHEDULES = 10
TOURNAMENT_SELECTION_SIZE = 5
MUTATION_RATE = 0.01
MAX_GENERATIONS = 100  # Define the maximum generations for termination
PENALTY_RATE = 0.01  # Define how much the penalty increases per generation
VARS = {'generationNum': 0,
        'terminateGens': False}

fitness_values = []

class Population:
    def __init__(self, size):
        self._size = size
        self._schedules = [Schedule().initialize() for _ in range(size)]

    def getSchedules(self):
        return self._schedules
    
    def __str__(self):
        schedule_descriptions = [str(schedule) for schedule in self._schedules]
        return (f"Population Size: {self._size}\n"
                f"Schedules: \n" + "\n".join(schedule_descriptions))
    


class Data:
    def __init__(self):
        self._rooms = Room.objects.all()
        self._meetingTimes = MeetingTime.objects.all()
        self._instructors = Instructor.objects.all()
        self._courses = Course.objects.all()
        self._depts = Department.objects.all()
        self._sections = Section.objects.all()

    def get_rooms(self):
        return self._rooms

    def get_instructors(self):
        return self._instructors

    def get_courses(self):
        return self._courses

    def get_depts(self):
        return self._depts

    def get_meetingTimes(self):
        return self._meetingTimes

    def get_sections(self):
        return self._sections


class Class:
    def __init__(self, dept, section, course):
        self.department = dept      # The department offering the class (e.g., Computer Science)
        self.course = course        # The specific course (e.g., "Data Structures")
        self.instructor = None      # Initially, there is no instructor assigned
        self.meeting_time = None    # Initially, there is no meeting time assigned
        self.room = None            # Initially, no room is assigned
        self.section = section 

class Schedule:
    def __init__(self):
        self._data = data
        self._classes = []
        self._numberOfConflicts = 0
        self._fitness = -1
        self._isFitnessChanged = True

    def getClasses(self):
        self._isFitnessChanged = True
        return self._classes

    def getNumbOfConflicts(self):
        return self._numberOfConflicts

    def getFitness(self):
        if self._isFitnessChanged:
            self._fitness = self.calculateFitness()
            self._isFitnessChanged = False
        return self._fitness

    def addCourse(self, data, course, courses, dept, section):
        newClass = Class(dept, section.section_id, course)

        newClass.set_meetingTime(
            data.get_meetingTimes()[random.randrange(0, len(data.get_meetingTimes()))])

        newClass.set_room(
            data.get_rooms()[random.randrange(0, len(data.get_rooms()))])

        crs_inst = course.instructors.all()
        newClass.set_instructor(
            crs_inst[random.randrange(0, len(crs_inst))])

        self._classes.append(newClass)

    def initialize(self):
        sections = Section.objects.all()

        for section in sections:
            dept = section.department
            n = section.num_class_in_week

            available_meeting_times = len(data.get_meetingTimes())
            if n > available_meeting_times:
                n = available_meeting_times

            courses = dept.courses.all()

            classes_to_add = n // len(courses)

            for course in courses:
                for i in range(classes_to_add):
                    self.addCourse(data, course, courses, dept, section)

            additional_classes = n % len(courses)

            for course in courses.order_by('?')[:additional_classes]:
                self.addCourse(data, course, courses, dept, section)

        return self

    def calculateFitness(self):
        self._numberOfHardConflicts = 0
        self._numberOfSoftConflicts = 0
        classes = self.getClasses()

        hard_weights = {
            'seating_capacity': 1,
            'same_course_same_section': 1.5,
            'instructor_conflict': 2,
            'duplicate_time_section': 2,
            'instructor_availability': 2,
        }

        soft_weights = {
            'no_consecutive_classes': 0.5,
            'morning_classes': 1,
            'break_time_conflict': 0.3,
            'balanced_days': 0.3
        }

        self.check_seating_capacity(classes, hard_weights)
        for i in range(len(classes)):
            self.check_course_conflicts(classes, i, hard_weights)
            self.check_instructor_conflict(classes, i, hard_weights)
            self.check_duplicate_time(classes, i, hard_weights)
            self.check_instructor_availability(classes, i, hard_weights)

        for i in range(len(classes)):
            self.check_consecutive_classes(classes, i, soft_weights)
            self.check_morning_classes(classes, i, soft_weights)
            self.check_break_time_conflict(classes, i, soft_weights)

        self.check_balanced_days(classes, soft_weights)

        total_conflict = (self._numberOfHardConflicts * 10) + self._numberOfSoftConflicts
        # Apply generation iteration penalty (using a simple linear increase in penalty)
        generation_penalty = VARS['generationNum'] * PENALTY_RATE
        total_conflict += generation_penalty  # Increase the conflict count with the penalty

        return 1 / (total_conflict + 1)  # Avoid division by zero


    # Check methods for hard and soft constraints (same as original)


class Particle:
    def __init__(self, schedule=None):
        self.schedule = schedule if schedule else Schedule().initialize()
        self.velocity = [random.uniform(-1, 1) for _ in range(len(self.schedule.getClasses()))]
        self.best_position = self.schedule
        self.best_fitness = self.get_fitness()

    def get_fitness(self):
        return self.schedule.getFitness()

    def update_position(self, global_best_position):
        for i in range(len(self.schedule.getClasses())):
            cognitive = C1 * random.random() * (self.best_position.getFitness() - self.schedule.getFitness())
            social = C2 * random.random() * (global_best_position.getFitness() - self.schedule.getFitness())
            
            self.velocity[i] = self.velocity[i] + cognitive + social
            
            current_class = self.schedule.getClasses()[i]
            current_class_position = current_class.get_position()
            new_position = current_class_position + self.velocity[i]
            current_class.set_position(new_position)

            if self.get_fitness() > self.best_fitness:
                self.best_position = self.schedule
                self.best_fitness = self.get_fitness()


class ParticleSwarmOptimization:
    def __init__(self):
        self.particles = [Particle() for _ in range(NUM_PARTICLES)]
        self.global_best_position = max(self.particles, key=lambda p: p.get_fitness()).best_position
        self.global_best_fitness = self.global_best_position.getFitness()

    def evolve(self):
        for particle in self.particles:
            particle.update_position(self.global_best_position)

        self.global_best_position = max(self.particles, key=lambda p: p.best_fitness).best_position
        self.global_best_fitness = self.global_best_position.getFitness()


@login_required
def timetable(request):
    global data
    data = Data()
    time_slots = TIME_SLOTS

    pso = ParticleSwarmOptimization()
    VARS['generationNum'] = 0
    VARS['terminateGens'] = False
    fitness_values = []

    while (pso.global_best_fitness != 1.0) and (VARS['generationNum'] <= MAX_GENERATIONS):
        if VARS['terminateGens']:
            return HttpResponse('')

        pso.evolve()
        VARS['generationNum'] += 1

        fitness_values.append(pso.global_best_fitness)
        print(f'\n> Generation #{VARS["generationNum"]}, Fitness: {pso.global_best_fitness}')

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(fitness_values)), fitness_values)
    plt.title('Fitness Over Generations')
    plt.xlabel('Generation Number')
    plt.ylabel('Fitness')
    plt.grid(True)

    plot_path = os.path.join(settings.MEDIA_ROOT, 'fitness_plot_pso.png')
    plt.savefig(plot_path)
    plt.close()

    break_time_slot = '10:00 - 10:50'
    week_days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']

    break_times = [(break_time_slot, day) for day in week_days]

    return render(
        request, 'timetable.html', {
            'schedule': pso.global_best_position.getClasses(),
            'sections': data.get_sections(),
            'times': data.get_meetingTimes(),
            'timeSlots': time_slots,
            'weekDays': DAYS_OF_WEEK,
            'break_times': break_times,
        })


def apiGenNum(request):
    return JsonResponse({'genNum': VARS['generationNum']})


def apiterminateGens(request):
    VARS['terminateGens'] = True
    return redirect('home')


def home(request):
    return render(request, 'index.html', {})


@login_required
def instructorAdd(request):
    form = InstructorForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('instructorAdd')
    context = {'form': form}
    return render(request, 'instructorAdd.html', context)


@login_required
def instructorEdit(request):
    context = {'instructors': Instructor.objects.all()}
    return render(request, 'instructorEdit.html', context)


@login_required
def instructorDelete(request, pk):
    inst = Instructor.objects.filter(pk=pk)
    if request.method == 'POST':
        inst.delete()
        return redirect('instructorEdit')


@login_required
def roomAdd(request):
    form = RoomForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('roomAdd')
    context = {'form': form}
    return render(request, 'roomAdd.html', context)


@login_required
def roomEdit(request):
    context = {'rooms': Room.objects.all()}
    return render(request, 'roomEdit.html', context)


@login_required
def roomDelete(request, pk):
    rm = Room.objects.filter(pk=pk)
    if request.method == 'POST':
        rm.delete()
        return redirect('roomEdit')


@login_required
def meetingTimeAdd(request):
    form = MeetingTimeForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('meetingTimeAdd')
        else:
            print('Invalid')
    context = {'form': form}
    return render(request, 'meetingTimeAdd.html', context)


@login_required
def meetingTimeEdit(request):
    context = {'meeting_times': MeetingTime.objects.all()}
    return render(request, 'meetingTimeEdit.html', context)


@login_required
def meetingTimeDelete(request, pk):
    mt = MeetingTime.objects.filter(pk=pk)
    if request.method == 'POST':
        mt.delete()
        return redirect('meetingTimeEdit')


@login_required
def courseAdd(request):
    form = CourseForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('courseAdd')
        else:
            print('Invalid')
    context = {'form': form}
    return render(request, 'courseAdd.html', context)


@login_required
def courseEdit(request):
    instructor = defaultdict(list)
    for course in Course.instructors.through.objects.all():
        course_number = course.course_id
        instructor_name = Instructor.objects.filter(
            id=course.instructor_id).values('name')[0]['name']
        instructor[course_number].append(instructor_name)

    context = {'courses': Course.objects.all(), 'instructor': instructor}
    return render(request, 'courseEdit.html', context)


@login_required
def courseDelete(request, pk):
    crs = Course.objects.filter(pk=pk)
    if request.method == 'POST':
        crs.delete()
        return redirect('courseEdit')


@login_required
def departmentAdd(request):
    form = DepartmentForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('departmentAdd')
    context = {'form': form}
    return render(request, 'departmentAdd.html', context)


@login_required
def departmentEdit(request):
    course = defaultdict(list)
    for dept in Department.courses.through.objects.all():
        dept_name = Department.objects.filter(
            id=dept.department_id).values('dept_name')[0]['dept_name']
        course_name = Course.objects.filter(
            course_number=dept.course_id).values(
                'course_name')[0]['course_name']
        course[dept_name].append(course_name)

    context = {'departments': Department.objects.all(), 'course': course}
    return render(request, 'departmentEdit.html', context)


@login_required
def departmentDelete(request, pk):
    dept = Department.objects.filter(pk=pk)
    if request.method == 'POST':
        dept.delete()
        return redirect('departmentEdit')


@login_required
def sectionAdd(request):
    form = SectionForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('sectionAdd')
    context = {'form': form}
    return render(request, 'sectionAdd.html', context)


@login_required
def sectionEdit(request):
    context = {'sections': Section.objects.all()}
    return render(request, 'sectionEdit.html', context)


@login_required
def sectionDelete(request, pk):
    sec = Section.objects.filter(pk=pk)
    if request.method == 'POST':
        sec.delete()
        return redirect('sectionEdit')




'''
Error pages
'''

def error_404(request, exception):
    return render(request,'errors/404.html', {})

def error_500(request, *args, **argv):
    return render(request,'errors/500.html', {})