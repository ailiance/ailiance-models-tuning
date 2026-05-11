#!/usr/bin/env python3
"""Expand ESP-IDF dataset with Espressif IoT Development Framework examples."""

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "processed" / "espidf_train.jsonl"
SYSTEM = "Tu es un expert en développement ESP32 avec ESP-IDF (Espressif IoT Development Framework). Réponds de manière technique et précise avec des exemples de code."

EXAMPLES = [
    ("Comment configurer le Wi-Fi en mode station avec ESP-IDF ?",
     """Configuration Wi-Fi STA avec ESP-IDF :

```c
#include "esp_wifi.h"
#include "esp_event.h"
#include "nvs_flash.h"

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();  // Auto-reconnect
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI("wifi", "IP: " IPSTR, IP2STR(&event->ip_info.ip));
    }
}

void wifi_init_sta(void) {
    nvs_flash_init();
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);

    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL);

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = "MonSSID",
            .password = "MonMotDePasse",
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();
}
```

**Points clés :**
- `nvs_flash_init()` obligatoire (stocke les calibrations PHY)
- Event loop asynchrone pour les événements Wi-Fi
- Auto-reconnexion dans le handler `WIFI_EVENT_STA_DISCONNECTED`"""),

    ("Comment utiliser MQTT avec ESP-IDF ?",
     """MQTT client avec ESP-IDF (composant `esp-mqtt`) :

```c
#include "mqtt_client.h"

static void mqtt_event_handler(void *arg, esp_event_base_t base,
                                int32_t event_id, void *event_data) {
    esp_mqtt_event_handle_t event = event_data;
    switch (event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI("mqtt", "Connected");
            esp_mqtt_client_subscribe(event->client, "/factory/status", 1);
            esp_mqtt_client_publish(event->client, "/factory/hello", "online", 0, 1, 0);
            break;
        case MQTT_EVENT_DATA:
            ESP_LOGI("mqtt", "Topic=%.*s Data=%.*s",
                     event->topic_len, event->topic,
                     event->data_len, event->data);
            break;
        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGW("mqtt", "Disconnected");
            break;
    }
}

void mqtt_app_start(void) {
    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = "mqtt://192.168.0.120:1883",
        .credentials.username = "device01",
    };
    esp_mqtt_client_handle_t client = esp_mqtt_client_init(&mqtt_cfg);
    esp_mqtt_client_register_event(client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    esp_mqtt_client_start(client);
}
```

**QoS :** 0 = fire-and-forget, 1 = at least once, 2 = exactly once
**TLS :** Ajouter `.broker.verification.certificate` pour le cert CA"""),

    ("Comment configurer le Bluetooth Low Energy (BLE) avec ESP-IDF ?",
     """BLE GATT Server avec ESP-IDF (NimBLE stack) :

```c
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

// UUID du service custom
static const ble_uuid128_t svc_uuid =
    BLE_UUID128_INIT(0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0,
                     0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0);

static int char_access_cb(uint16_t conn_handle, uint16_t attr_handle,
                          struct ble_gatt_access_ctxt *ctxt, void *arg) {
    if (ctxt->op == BLE_GATT_ACCESS_OP_READ_CHR) {
        const char *value = "Hello BLE";
        os_mbuf_append(ctxt->om, value, strlen(value));
    }
    return 0;
}

static const struct ble_gatt_svc_def gatt_svr_svcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &svc_uuid.u,
        .characteristics = (struct ble_gatt_chr_def[]) {
            {
                .uuid = BLE_UUID16_DECLARE(0xFF01),
                .access_cb = char_access_cb,
                .flags = BLE_GATT_CHR_F_READ,
            }, { 0 }
        },
    }, { 0 }
};

void ble_init(void) {
    nimble_port_init();
    ble_svc_gap_device_name_set("Ailiance-Device");
    ble_svc_gap_init();
    ble_svc_gatt_init();
    ble_gatts_count_cfg(gatt_svr_svcs);
    ble_gatts_add_svcs(gatt_svr_svcs);
    nimble_port_freertos_init(ble_host_task);
}
```

**NimBLE vs Bluedroid :** NimBLE = moins de RAM (~50KB vs ~200KB), recommandé pour ESP32-C3/S3"""),

    ("Comment utiliser le système de partitions et l'OTA sur ESP32 ?",
     """Partitions + OTA avec ESP-IDF :

**1. Table de partitions (`partitions.csv`) :**
```csv
# Name,   Type, SubType, Offset,  Size
nvs,      data, nvs,     0x9000,  0x4000
otadata,  data, ota,     0xd000,  0x2000
phy_init, data, phy,     0xf000,  0x1000
ota_0,    app,  ota_0,   0x10000, 0x1E0000
ota_1,    app,  ota_1,   0x1F0000,0x1E0000
```

**2. OTA via HTTPS :**
```c
#include "esp_ota_ops.h"
#include "esp_http_client.h"
#include "esp_https_ota.h"

void ota_update(const char *url) {
    esp_http_client_config_t config = {
        .url = url,
        .cert_pem = server_cert_pem,  // CA certificate
    };

    esp_https_ota_config_t ota_config = {
        .http_config = &config,
    };

    esp_err_t ret = esp_https_ota(&ota_config);
    if (ret == ESP_OK) {
        ESP_LOGI("ota", "OTA success, restarting...");
        esp_restart();
    } else {
        ESP_LOGE("ota", "OTA failed: %s", esp_err_to_name(ret));
    }
}
```

**Rollback automatique :**
```c
// Au boot, valider le firmware si tout va bien
esp_ota_mark_app_valid_cancel_rollback();
// Si le firmware crashe, l'ESP32 revient automatiquement à l'ancienne version
```"""),

    ("Comment utiliser les tâches FreeRTOS avec ESP-IDF ?",
     """FreeRTOS sur ESP-IDF (dual-core ESP32) :

```c
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"

static QueueHandle_t sensor_queue;
static SemaphoreHandle_t spi_mutex;

// Tâche capteur — tourne sur Core 0
void sensor_task(void *pvParam) {
    float temperature;
    while (1) {
        temperature = read_temperature_sensor();
        xQueueSend(sensor_queue, &temperature, portMAX_DELAY);
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

// Tâche réseau — tourne sur Core 1
void network_task(void *pvParam) {
    float temp;
    while (1) {
        if (xQueueReceive(sensor_queue, &temp, pdMS_TO_TICKS(5000))) {
            mqtt_publish_temperature(temp);
        }
    }
}

void app_main(void) {
    sensor_queue = xQueueCreate(10, sizeof(float));
    spi_mutex = xSemaphoreCreateMutex();

    // Pin task to specific core
    xTaskCreatePinnedToCore(sensor_task, "sensor", 4096, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(network_task, "network", 8192, NULL, 3, NULL, 1);
}
```

**Bonnes pratiques ESP-IDF :**
- Stack minimum : 2048 pour tâches simples, 8192+ pour réseau/TLS
- `app_main` tourne sur Core 0 par défaut
- Utiliser `xTaskCreatePinnedToCore` pour contrôler l'affinité
- Priority : 0 = idle, 5 = normal, 24 = max (configMAX_PRIORITIES)"""),

    ("Comment utiliser le deep sleep et le wake-up sur ESP32 ?",
     """Deep sleep avec ESP-IDF :

```c
#include "esp_sleep.h"
#include "driver/rtc_io.h"

void enter_deep_sleep(void) {
    // Méthode 1 : Timer wake-up (10 secondes)
    esp_sleep_enable_timer_wakeup(10 * 1000000);  // µs

    // Méthode 2 : GPIO wake-up (bouton sur GPIO 33)
    esp_sleep_enable_ext0_wakeup(GPIO_NUM_33, 0);  // 0 = LOW level

    // Méthode 3 : Touch pad wake-up
    esp_sleep_enable_touchpad_wakeup();

    // Désactiver les périphériques inutiles
    esp_wifi_stop();
    esp_bt_controller_disable();

    // Conserver des données en RTC memory
    RTC_DATA_ATTR static int boot_count = 0;
    boot_count++;
    ESP_LOGI("sleep", "Boot count: %d", boot_count);

    // Dormir
    ESP_LOGI("sleep", "Entering deep sleep...");
    esp_deep_sleep_start();
}

void app_main(void) {
    esp_sleep_wakeup_cause_t cause = esp_sleep_get_wakeup_cause();
    switch (cause) {
        case ESP_SLEEP_WAKEUP_TIMER:
            ESP_LOGI("main", "Wakeup: timer"); break;
        case ESP_SLEEP_WAKEUP_EXT0:
            ESP_LOGI("main", "Wakeup: GPIO"); break;
        default:
            ESP_LOGI("main", "Wakeup: power on"); break;
    }
    // ... faire les mesures, envoyer les données ...
    enter_deep_sleep();
}
```

**Consommation :** Active ~160mA, Light sleep ~0.8mA, Deep sleep ~10µA, Hibernation ~5µA"""),

    ("Comment utiliser le système de logs et le monitoring avec ESP-IDF ?",
     """Logging et monitoring ESP-IDF :

```c
// Niveaux de log
ESP_LOGE("TAG", "Error");     // Rouge
ESP_LOGW("TAG", "Warning");   // Jaune
ESP_LOGI("TAG", "Info");      // Vert
ESP_LOGD("TAG", "Debug");     // Pas affiché par défaut
ESP_LOGV("TAG", "Verbose");   // Pas affiché par défaut

// Configurer le niveau par tag dans menuconfig
// ou dynamiquement :
esp_log_level_set("wifi", ESP_LOG_WARN);
esp_log_level_set("*", ESP_LOG_INFO);  // Tous les tags

// Heap monitoring
ESP_LOGI("mem", "Free heap: %lu", esp_get_free_heap_size());
ESP_LOGI("mem", "Min free: %lu", esp_get_minimum_free_heap_size());
ESP_LOGI("mem", "PSRAM free: %lu", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

// Task monitoring
void print_task_stats(void) {
    char buf[2048];
    vTaskList(buf);
    ESP_LOGI("tasks", "\\nName\\t\\tState\\tPrio\\tStack\\tNum\\n%s", buf);

    vTaskGetRunTimeStats(buf);
    ESP_LOGI("tasks", "\\nName\\t\\tTime\\t\\t%%\\n%s", buf);
}

// Panic handler personnalisé
void esp_custom_panic_handler(esp_panic_info_t *info) {
    // Sauvegarder le crash log en NVS avant reboot
    nvs_flash_init();
    // ... écrire info->reason, info->addr ...
}
```

**`menuconfig` important :**
- `CONFIG_LOG_DEFAULT_LEVEL` : niveau de log par défaut
- `CONFIG_ESP_SYSTEM_PANIC` : `print_regs_and_halt` pour debug
- `CONFIG_FREERTOS_USE_TRACE_FACILITY` : activer pour `vTaskList`"""),

    ("Comment configurer l'I2S pour l'audio sur ESP32 ?",
     """I2S Audio avec ESP-IDF (nouveau driver v5+) :

```c
#include "driver/i2s_std.h"

i2s_chan_handle_t tx_handle;

void i2s_init(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    i2s_new_channel(&chan_cfg, &tx_handle, NULL);

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(44100),  // 44.1 kHz
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .bclk = GPIO_NUM_26,
            .ws = GPIO_NUM_25,
            .dout = GPIO_NUM_22,
            .din = I2S_GPIO_UNUSED,
        },
    };
    i2s_channel_init_std_mode(tx_handle, &std_cfg);
    i2s_channel_enable(tx_handle);
}

// Jouer un buffer audio
void play_audio(const int16_t *samples, size_t len) {
    size_t bytes_written;
    i2s_channel_write(tx_handle, samples, len * sizeof(int16_t),
                      &bytes_written, portMAX_DELAY);
}

// Générer un tone 440 Hz
void play_tone(float freq, float duration_s) {
    const int sample_rate = 44100;
    int num_samples = (int)(sample_rate * duration_s);
    int16_t *buf = malloc(num_samples * 2 * sizeof(int16_t));  // stereo

    for (int i = 0; i < num_samples; i++) {
        int16_t sample = (int16_t)(32767.0f * sinf(2.0f * M_PI * freq * i / sample_rate));
        buf[i * 2] = sample;      // Left
        buf[i * 2 + 1] = sample;  // Right
    }
    play_audio(buf, num_samples * 2);
    free(buf);
}
```

**DACs supportés :** MAX98357A (I2S), PCM5102, ES8388, WM8978
**Codecs :** MP3 via `esp-adf` ou `libhelix-mp3`"""),
]


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            existing = [json.loads(line) for line in f if line.strip()]

    new_examples = []
    for i, (q, a) in enumerate(EXAMPLES):
        new_examples.append({
            "id": f"espidf-expanded-{i:04d}",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
        })

    all_ex = existing + new_examples
    with open(OUTPUT, "w") as f:
        for ex in all_ex:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"ESP-IDF dataset: {len(existing)} existing + {len(new_examples)} new = {len(all_ex)} total")


if __name__ == "__main__":
    build()
