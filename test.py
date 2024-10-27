import pygame, sys, random
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def dist(l1, l2):
    return (((l2[0] - l1[0]) ** 2) + ((l2[1] - l1[1]) ** 2)) ** 0.5


def finger(handlandmarks, shape=(1, 1)):
    try:

        needful = [handlandmarks.landmark[0].x * shape[0], handlandmarks.landmark[0].y * shape[1]]
        d07 = dist(needful, [handlandmarks.landmark[7].x * shape[0],
                             handlandmarks.landmark[7].y * shape[1]])
        d08 = dist(needful, [handlandmarks.landmark[8].x * shape[0],
                             handlandmarks.landmark[8].y * shape[1]])
        d010 = dist(needful, [handlandmarks.landmark[10].x * shape[0],
                              handlandmarks.landmark[10].y * shape[1]])
        d012 = dist(needful, [handlandmarks.landmark[12].x * shape[0],
                              handlandmarks.landmark[12].y * shape[1]])
        d014 = dist(needful, [handlandmarks.landmark[14].x * shape[0],
                              handlandmarks.landmark[14].y * shape[1]])
        d016 = dist(needful, [handlandmarks.landmark[16].x * shape[0],
                              handlandmarks.landmark[16].y * shape[1]])
        d018 = dist(needful, [handlandmarks.landmark[18].x * shape[0],
                              handlandmarks.landmark[18].y * shape[1]])
        d020 = dist(needful, [handlandmarks.landmark[20].x * shape[0],
                              handlandmarks.landmark[20].y * shape[1]])

        closed = []
        if d07 > d08:
            closed.append(1)
        if d010 > d012:
            closed.append(2)
        if d014 > d016:
            closed.append(3)
        if d018 > d020:
            closed.append(4)

        return closed
    except:
        pass


def draw_floor():
    screen.blit(floor_surface, (floor_x_position, 900))
    screen.blit(floor_surface, (floor_x_position + 576, 900))


def create_pipe():
    random_pipe_pos = random.choice(pipe_hights)
    bottom_pipe = pipe_surface.get_rect(midtop=(700, random_pipe_pos))
    top_pipe = pipe_surface.get_rect(midbottom=(700, random_pipe_pos - 350))
    return bottom_pipe, top_pipe


def move_pipes(pipes):
    for pipe in pipes:
        pipe.centerx -= 10
    return pipes


def draw_pipes(pipes):
    for pipe in pipes:
        if pipe.bottom >= 1024:
            screen.blit(pipe_surface, pipe)
        else:
            flip_pipe = pygame.transform.flip(pipe_surface, False, True)
            screen.blit(flip_pipe, pipe)


def check_collision(pipes):
    for pipe in pipes:
        if bird_rect.colliderect(pipe):
            # death_sound.play()
            return False
        if bird_rect.top <= -100 or bird_rect.bottom >= 900:
            # death_sound.play(  )
            return False
    return True


def rotate_bird(bird):
    return pygame.transform.rotozoom(bird, -bird_movement, 1)


def bird_animation():
    new_bird = bird_frames[bird_index]
    new_bird_rect = new_bird.get_rect(center=(100, bird_rect.centery))
    return new_bird, new_bird_rect


def score_display(game_state):
    if game_state == "game over":
        score_surface = game_font.render(f"Score: {int(score)}", True, (255, 255, 255))
        score_rect = score_surface.get_rect(center=(288, 100))
        screen.blit(score_surface, score_rect)

        high_score_surface = game_font.render(f"High score: {int(high_score)}", True, (255, 255, 255))
        high_score_rect = high_score_surface.get_rect(center=(288, 850))
        screen.blit(high_score_surface, high_score_rect)

    if game_state == "main game":
        score_surface = game_font.render(f"Score: {int(score)}", True, (255, 255, 255))
        score_rect = score_surface.get_rect(center=(288, 100))
        screen.blit(score_surface, score_rect)


def update_score(score, high_score):
    if score > high_score:
        high_score = score
    return high_score


pygame.mixer.pre_init()
pygame.init()
screen = pygame.display.set_mode((576, 1024))
clock = pygame.time.Clock()
game_font = pygame.font.Font("C:\\Users\Monika\PycharmProjects\Testing\\04B_19.TTF", 40)

# Game Variables
gravity = 0.75
bird_movement = 0
game_active = True
score = 0
high_score = 0

bg_surface = pygame.image.load("C:\\Users\Monika\PycharmProjects\Testing\\background-day.png").convert()
bg_surface = pygame.transform.scale2x(bg_surface)

game_over_surface = pygame.image.load("C:\\Users\Monika\PycharmProjects\Testing\\message.png").convert_alpha()
game_over_surface = pygame.transform.scale2x(game_over_surface)
game_over_rect = game_over_surface.get_rect(center=(288, 512))

floor_surface = pygame.image.load("C:\\Users\Monika\PycharmProjects\Testing\\base.png").convert()
floor_surface = pygame.transform.scale2x(floor_surface)
floor_x_position = 0

bird_downflap = pygame.image.load("C:\\Users\Monika\PycharmProjects\Testing\\redbird-downflap.png").convert_alpha()
bird_downflap = pygame.transform.scale2x(bird_downflap)
bird_midflap = pygame.image.load("C:\\Users\Monika\PycharmProjects\Testing\\redbird-midflap.png").convert_alpha()
bird_midflap = pygame.transform.scale2x(bird_midflap)
bird_upflap = pygame.image.load("C:\\Users\Monika\PycharmProjects\Testing\\redbird-upflap.png").convert_alpha()
bird_upflap = pygame.transform.scale2x(bird_upflap)
bird_frames = [bird_downflap, bird_midflap, bird_upflap]
bird_index = 0
bird_surface = bird_frames[bird_index]
bird_rect = bird_surface.get_rect(center=(100, 512))

BIRDFLAP = pygame.USEREVENT + 1
pygame.time.set_timer(BIRDFLAP, 200)

pipe_surface = pygame.image.load("C:\\Users\Monika\PycharmProjects\Testing\\pipe-green.png").convert()
pipe_surface = pygame.transform.scale2x(pipe_surface)
pipe_list = []
SPAWNPIPE = pygame.USEREVENT
pygame.time.set_timer(SPAWNPIPE, 6000)
pipe_hights = [400, 600, 800]

flap_sound = pygame.mixer.Sound("C:\\Users\Monika\PycharmProjects\Testing\\sound_sfx_wing.wav")
death_sound = pygame.mixer.Sound("C:\\Users\Monika\PycharmProjects\Testing\\sound_sfx_hit.wav")
score_sound = pygame.mixer.Sound("C:\\Users\Monika\PycharmProjects\Testing\\sound_sfx_point.wav")

num_of_frames = -1
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.8,
                    min_tracking_confidence=0.8, max_num_hands=1) as hands:
    # Game LOOP
    while True and cap.isOpened():
        num_of_frames += 1
        if num_of_frames % 1 == 0:
            success, img = cap.read()
            image = img.copy()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            frame_shape = (image.shape[1], image.shape[0])

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    closed = finger(hand_landmarks, frame_shape)
                    if closed == [1, 2, 3, 4] or closed == [1, 2, 3] or closed == [2, 3, 4]:
                        camera_event = "K_space"

            # Checking for Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == SPAWNPIPE:
                    pipe_list.extend(create_pipe())

                if event.type == BIRDFLAP:
                    if bird_index < 2:
                        bird_index += 1
                    else:
                        bird_index = 0

                    bird_surface, bird_rect = bird_animation()

            try:
                if camera_event == "K_space" and game_active:
                    # bird_movement = 0
                    bird_movement = -10
                    # flap_sound.play()
                if camera_event == "K_space" and not (game_active):
                    game_active = True
                    pipe_list.clear()
                    bird_rect.center = (100, 512)
                    bird_movement = 0
                    score = 0
            except:
                pass

            # Drawing Backgound
            screen.blit(bg_surface, (0, 0))

            if game_active:
                # Drawing The Bird
                # Bird Movement
                bird_movement += gravity
                rotated_bird = rotate_bird(bird_surface)
                bird_rect.centery += bird_movement
                screen.blit(rotated_bird, bird_rect)
                game_active = check_collision(pipe_list)

                # Dealing with Pipes
                pipe_list = move_pipes(pipe_list)
                draw_pipes(pipe_list)

                score += 0.05
                score_display("main game")
            else:
                screen.blit(game_over_surface, game_over_rect)
                high_score = update_score(score, high_score)
                score_display("game over")

            # Floor Movement and Drawing
            floor_x_position -= 8
            draw_floor()
            if floor_x_position <= -576:
                floor_x_position = 0

            # Showing Updates and Limiting our Frames per Second
            pygame.display.update()
            clock.tick(100)
            camera_event = None

            cv2.imshow('MediaPipe Hands', image)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                pygame.quit()
                break

cap.release()
cv2.destroyAllWindows()
