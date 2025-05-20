#define cooling_AOM (1 << PB1)
#define repumping_AOM (1 << PB2)
#define magnetic_field (1 << PB3)
#define imaging_AOM (1 << PB4)
#define current_enable (1 << PB5)

void setup()
{
    DDRB |= cooling_AOM | repumping_AOM | magnetic_field | imaging_AOM | current_enable;
}

void loop()
{
    PORTB |= cooling_AOM | repumping_AOM;

    PORTB |= magnetic_field;

    delay(500);

    PORTB &= ~magnetic_field;

    PORTB &= ~(cooling_AOM | repumping_AOM);

    delay(100);

    PORTB |= imaging_AOM;

    delay(10);

    PORTB &= ~imaging_AOM;

    delay(50);
}
