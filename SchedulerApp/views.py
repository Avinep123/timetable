from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import *
from .forms import *
import random
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from django.conf import settings
from collections import defaultdict

w=0.5
# Constants for PSO
C1 = 2  # Cognitive component
C2 = 2  # Social component
TIMESLOTS_PER_DAY = 5
NUM_PARTICLES = 200
MAX_GENERATIONS = 1000
VARS = {'generationNum': 0, 'terminateGens': False}
fitness_values = []

# Data class to fetch all data from the models
class Data:
    def __init__(self):
        self._rooms = list(Room.objects.all())
        self._meetingTimes = list(MeetingTime.objects.all())
        self._instructors = list(Instructor.objects.all())
        self._courses = list(Course.objects.all())
        self._depts = list(Department.objects.all())
        self._sections = list(Section.objects.all())

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
        self.department = dept
        self.course = course  # course should be an object or value
        self.instructor = None
        self.meeting_time = None
        self.room = None
        self.section = section

    def set_instructor(self, instructor):
        self.instructor = instructor

    def get_instructor(self):
        return self.instructor

    def set_meetingTime(self, meetingTime):
        self.meeting_time = meetingTime

    def get_meetingTime(self):
        return self.meeting_time

    def set_room(self, room):
        self.room = room

    def get_room(self):
        return self.room

    def get_course(self):
        return self.course  # Return the course object or its details

    def get_position(self):
        return (self.room, self.meeting_time, self.instructor)

    def set_position(self, position):
        self.room, self.meeting_time, self.instructor = position

    def __str__(self):
        return f"Class(dept={self.department}, course={self.course}, section={self.section}, instructor={self.instructor}, meeting_time={self.meeting_time}, room={self.room})"



# Schedule class to calculate fitness and manage course placement
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
        newClass = Class(dept, section.section_id, course)  # Ensure `course` is passed correctly
        newClass.set_meetingTime(data.get_meetingTimes()[random.randrange(0, len(data.get_meetingTimes()))])
        newClass.set_room(data.get_rooms()[random.randrange(0, len(data.get_rooms()))])
        crs_inst = course.instructors.all()
        newClass.set_instructor(crs_inst[random.randrange(0, len(crs_inst))])
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


    from collections import defaultdict

    
    def calculateFitness(self):
        conflicts = 0
        penalties = {
            "seating_capacity": 10,
            "same_course_same_section": 10,
            "instructor_conflict": 20,
            "duplicate_time_section": 20,
            "instructor_availability": 20,
            "empty_timeslot": 20
        }

        max_possible_conflicts = sum(penalties.values())

        instructor_time_conflict = {}
        room_time_conflict = {}
        section_timeslot_conflict = {}
        timeslot_occupancy = defaultdict(int)

        for c in self.getClasses():
            timeslot_id = c.get_meetingTime().pid
            instructor_id = c.get_instructor().id
            room_id = c.get_room().id
            section_id = c.section

            timeslot_occupancy[timeslot_id] += 1

            conflict_penalty = self.check_and_add_penalties(
                instructor_time_conflict, instructor_id, timeslot_id, penalties["instructor_conflict"],
                room_time_conflict, room_id, timeslot_id, penalties["seating_capacity"],
                section_timeslot_conflict, section_id, timeslot_id, penalties["same_course_same_section"]
            )

            conflicts += conflict_penalty

        for timeslot in data.get_meetingTimes():
            if timeslot_occupancy[timeslot.pid] == 0:
                conflicts += penalties["empty_timeslot"]

        fitness = 1 - (conflicts / max_possible_conflicts)

        print(f"Total penalty for this generation: {conflicts}")
        self._numberOfConflicts = conflicts
        return fitness

    def check_and_add_penalties(self, instructor_time_conflict, instructor_id, timeslot_id, instructor_penalty,
                                room_time_conflict, room_id, timeslot_id_room, room_penalty,
                                section_timeslot_conflict, section_id, timeslot_id_section, section_penalty):
        conflicts = 0

        if timeslot_id in instructor_time_conflict and instructor_id in instructor_time_conflict[timeslot_id]:
            conflicts += instructor_penalty
        else:
            instructor_time_conflict.setdefault(timeslot_id, []).append(instructor_id)

        if timeslot_id in room_time_conflict and room_id in room_time_conflict[timeslot_id]:
            conflicts += room_penalty
        else:
            room_time_conflict.setdefault(timeslot_id, []).append(room_id)

        if section_id in section_timeslot_conflict and timeslot_id in section_timeslot_conflict[section_id]:
            conflicts += section_penalty
        else:
            section_timeslot_conflict.setdefault(section_id, []).append(timeslot_id)

        return conflicts

  







    def __str__(self):
        return f"Schedule with {len(self._classes)} classes."

# Particle class representing each particle in the PSO algorithm
# Particle class representing each particle in the PSO algorithm
class Particle:
    def __init__(self, schedule=None):
        self.schedule = schedule if schedule else Schedule().initialize()
        self._isFitnessChanged = True
        self.velocity = [random.uniform(-1, 1) for _ in range(len(self.schedule.getClasses()))]
        self.best_position = self.schedule
        self.best_fitness = self.getFitness()

    def getFitness(self):
        return self.schedule.getFitness()

    def update_position(self, global_best_position):
        for i in range(len(self.schedule.getClasses())):
            cognitive = C1 * random.random() * (self.best_position.getFitness() - self.schedule.getFitness())
            social = C2 * random.random() * (global_best_position.getFitness() - self.schedule.getFitness())

            # Update velocity with inertia weight
            self.velocity[i] = w * self.velocity[i] + cognitive + social
            self.velocity[i] = np.clip(self.velocity[i], -1, 1)  # Ensure velocity remains within bounds

            current_class = self.schedule.getClasses()[i]
            new_meeting_time = data.get_meetingTimes()[random.randint(0, len(data.get_meetingTimes()) - 1)]
            new_room = data.get_rooms()[random.randint(0, len(data.get_rooms()) - 1)]
            new_instructor = current_class.get_instructor()

            new_position = (new_room, new_meeting_time, new_instructor)
            current_class.set_position(new_position)

            if self.getFitness() > self.best_fitness:
                self.best_position = self.schedule
                self.best_fitness = self.getFitness()


# Main PSO optimization class
class ParticleSwarmOptimization:
    def __init__(self):
        self.particles = [Particle() for _ in range(NUM_PARTICLES)]
        self.global_best_position = max(self.particles, key=lambda p: p.getFitness()).best_position
        self.global_best_fitness = self.global_best_position.getFitness()

    def evolve(self):
        for particle in self.particles:
            particle.update_position(self.global_best_position)

        self.global_best_position = max(self.particles, key=lambda p: p.best_fitness).best_position
        self.global_best_fitness = self.global_best_position.getFitness()

# Django view for timetable
data = None
@login_required
def timetable(request):
    global data
    if not data:  # Initialize if it's not already initialized
        data = Data()


    pso = ParticleSwarmOptimization()
    VARS['generationNum'] = 0
    VARS['terminateGens'] = False

    fitness_values = []
    penalties = []

    # Run the PSO algorithm until a solution is found or max generations are reached
    while (pso.global_best_fitness != 1.0) and (VARS['generationNum'] <= MAX_GENERATIONS):
        if VARS['terminateGens']:
            return HttpResponse('')

        # Evolve the particles in the current generation
        pso.evolve()

        VARS['generationNum'] += 1
        current_fitness = pso.global_best_fitness
        current_penalty = pso.global_best_position.getNumbOfConflicts()

        # Track fitness and penalty values for every generation
        fitness_values.append(current_fitness)
        penalties.append(current_penalty)

        # Print fitness and penalty values for each generation
        print(f"Generation {VARS['generationNum']}: Fitness = {current_fitness}, Penalty = {current_penalty}")

    # After the loop ends, plot fitness vs. penalty over generations
    print(f"Fitness values: {fitness_values}")
    print(f"Penalties: {penalties}")

    # Plot fitness vs. penalty over generations
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(fitness_values)), fitness_values, label='Fitness', color='b')
    plt.plot(range(len(penalties)), penalties, label='Penalty', color='r', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.title('Fitness vs Penalty Over Generations')
    plt.legend()

    plot_path = os.path.join(settings.MEDIA_ROOT, 'fitness_vs_penalty.png')
    plt.savefig(plot_path)

    plt.close()
    break_time_slot = '10:00 - 10:50'  # The break time you want to use
    week_days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']  # List of weekdays

    # Generate break times for all weekdays
    break_times = [(break_time_slot, day) for day in week_days]

    # Render the timetable with the generated schedule and plot
    return render(
        request, 'timetable.html', {
            'schedule': pso.global_best_position.getClasses(),
            'sections': data.get_sections(),
            'times': data.get_meetingTimes(),
            'timeSlots': TIME_SLOTS,
            'weekDays': DAYS_OF_WEEK,
            'break_times': break_times,
            'plot_path': plot_path
        }
    )



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
