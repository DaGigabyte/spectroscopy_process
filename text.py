import turtle
import time
font_size = 150
turtle.penup()
turtle.hideturtle()
turtle.sety(-font_size/2)
turtle.tracer(False)
while(True):
    turtle.clear()
    turtle.write('Hello', align='center', font=("Verdana", font_size, "normal"))
    time.sleep(1)
    turtle.clear()
    turtle.write('world', align='center', font=("Verdana", font_size, "normal"))
    time.sleep(1)
    turtle.update()
turtle.done()