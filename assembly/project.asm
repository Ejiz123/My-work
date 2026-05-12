org 100h

jmp start

menu        db 0Dh,0Ah,'Select Difficulty Level:$'
easymsg     db 0Dh,0Ah,'1. Easy (0-9)$'
mediummsg   db 0Dh,0Ah,'2. Medium (0-50)$'
hardmsg     db 0Dh,0Ah,'3. Hard (0-99)$'
choicemsg   db 0Dh,0Ah,'Enter choice: $'

msg1        db 0Dh,0Ah,'Guess the number: $'
high        db 0Dh,0Ah,'Too High! Try again.',0Dh,0Ah,'$'

low         db 0Dh,0Ah,'Too Low! Try again.',0Dh,0Ah,'$'

win         db 0Dh,0Ah,'Correct Guess! You Win.',0Dh,0Ah,'$'

invalid     db 0Dh,0Ah,'Invalid input!',0Dh,0Ah,'$'

gameover    db 0Dh,0Ah,'Game Over! No attempts left.',0Dh,0Ah,'$'
againmsg    db 0Dh,0Ah,'Play Again? (Y/N): $'

secret      db 0
attempts    db 5
maxnum      db 9

guess       db 0

start:

main_menu:

    ; Display menu
    mov ah,09h
    mov dx,menu
    int 21h

    mov dx,easymsg
    int 21h

    mov dx,mediummsg
    int 21h

    mov dx,hardmsg
    int 21h

    mov dx,choicemsg
    int 21h

    ; Input choice
    mov ah,01h
    int 21h

    cmp al,'1'
    je easy

    cmp al,'2'
    je medium

    cmp al,'3'
    je hard

    jmp main_menu


easy:
    mov byte [maxnum],9
    mov byte [attempts],5
    jmp generate_random


medium:
    mov byte [maxnum],50
    mov byte [attempts],7
    jmp generate_random


hard:
    mov byte [maxnum],99
    mov byte [attempts],10


generate_random:

    mov ah,2Ch
    int 21h

    mov al,dl
    mov ah,0

    mov bl,[maxnum]
    inc bl

    div bl

    mov [secret],ah

game_loop:

    ; Check attempts
    cmp byte [attempts],0
    je game_over_screen

    ; Show input message
    mov ah,09h
    mov dx,msg1
    int 21h

    ; Input first digit
    mov ah,01h
    int 21h

    ; Validate first digit
    cmp al,'0'
    jb invalid_input

    cmp al,'9'
    ja invalid_input

    sub al,'0'

    mov bl,10
    mul bl

    mov bh,al

    ; Input second digit
    mov ah,01h
    int 21h

    cmp al,13
    je single_digit

    cmp al,'0'
    jb invalid_input

    cmp al,'9'
    ja invalid_input

    sub al,'0'

    add bh,al

    mov [guess],bh
    jmp compare_number

single_digit:

    mov [guess],bh
    mov al,[guess]
    mov bl,10
    div bl
    mov [guess],al


compare_number:

    mov al,[guess]
    mov bl,[secret]

    cmp al,bl
    je correct

    ja too_high
    jb too_low

too_high:

    dec byte [attempts]

    mov ah,09h
    mov dx,high
    int 21h

    jmp game_loop

too_low:

    dec byte [attempts]

    mov ah,09h
    mov dx,low
    int 21h

    jmp game_loop


invalid_input:

    mov ah,09h
    mov dx,invalid
    int 21h

    jmp game_loop

correct:

    mov ah,09h
    mov dx,win
    int 21h

    jmp play_again

game_over_screen:

    mov ah,09h
    mov dx,gameover
    int 21h

play_again:

    mov ah,09h
    mov dx,againmsg
    int 21h

    mov ah,01h
    int 21h

    cmp al,'Y'
    je start

    cmp al,'y'
    je start

    ; Exit Program
    mov ah,4Ch
    int 21h