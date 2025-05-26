#define current_enable (1 << PB1)
#define cooling_AOM (1 << PB2)
#define repump_AOM (1 << PB3)
#define imaging_AOM (1 << PB4)
#define cam (1 << PB5)
#define sequence_trigger (1 << PD3)

void setup()
{
    pinMode(13, OUTPUT);
    pinMode(12, OUTPUT);
    pinMode(11, OUTPUT);
    pinMode(10, OUTPUT);
    pinMode(9, OUTPUT);

    Serial.begin(9600);
    PORTB &= ~(current_enable | cooling_AOM | repump_AOM | imaging_AOM | cam);
}

void loop()
{
    for (int test = 71; test <= 5000; test += 250)
    {
        int tof = test; // us
        Serial.print("tof: ");
        Serial.println(tof);
        int delay_between_images = 30; // ms
        int exposure_duration = 20;    // us
        int camera_delay = 50;         // us
        // switch on the MOT
        PORTB |= current_enable | cooling_AOM | repump_AOM;
        delay(15000); // load the MOT
        PORTB &= ~(current_enable | cooling_AOM | repump_AOM);

        delayMicroseconds(tof - camera_delay);
        // take the first image
        PORTB |= cam;
        delayMicroseconds(camera_delay);
        PORTB |= imaging_AOM;
        delayMicroseconds(exposure_duration);
        PORTB &= ~(imaging_AOM | cam);
        delay(delay_between_images);

        // take the second image
        PORTB |= cam;
        delayMicroseconds(camera_delay);
        PORTB |= imaging_AOM;
        delayMicroseconds(exposure_duration);
        PORTB &= ~(imaging_AOM | cam);
        delay(delay_between_images);

        // take the third image
        PORTB |= cam;
        delayMicroseconds(camera_delay);
        // no light to get a background
        delayMicroseconds(exposure_duration);
        PORTB &= ~(imaging_AOM | cam);
        delay(delay_between_images);
        delay(10);
    }
}