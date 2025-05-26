#define current_enable (1 << PB1)
#define cooling_AOM (1 << PB2)
#define repump_AOM (1 << PB3)
#define trigger (1 << PB4)

void setup()
{
    DDRB |= cooling_AOM | current_enable | repump_AOM | trigger;
}

void loop()
{
    PORTB |= cooling_AOM | current_enable | repump_AOM;

    delay(15000);

    PORTB &= ~(current_enable | cooling_AOM | repump_AOM);

    delay(50);

    PORTB |= cooling_AOM | current_enable | repump_AOM | trigger;

    delay(100);

    PORTB &= ~(current_enable | cooling_AOM | repump_AOM | trigger);

    delay(200);
}