"use strict";

const DEFAULTS = {
  tellers: 4,
  hours: 8,
  arrivalRate: 20,
  serviceRate: 24,
  paymentProb: 0.55,
  replications: 10,
  relativeError: 0.05,
  waitThreshold: 5,
  maxReplications: 500,
  seed: null,
};

const KEY_MAP = {
  tellers: "tellers",
  hours: "hours",
  "arrival-rate": "arrivalRate",
  "service-rate": "serviceRate",
  "payment-prob": "paymentProb",
  replications: "replications",
  "relative-error": "relativeError",
  "wait-threshold": "waitThreshold",
  "max-replications": "maxReplications",
  seed: "seed",
};

const Z_SCORE_95_CI = 1.96;

function usage() {
  return `
Uso:
  node src/index.js [opciones]

Opciones:
  --tellers <n>           Número de cajeros (default: ${DEFAULTS.tellers})
  --hours <n>             Horas de simulación (default: ${DEFAULTS.hours})
  --arrival-rate <n>      Tasa de llegadas por hora (default: ${DEFAULTS.arrivalRate})
  --service-rate <n>      Tasa de servicio por hora (default: ${DEFAULTS.serviceRate})
  --payment-prob <n>      Probabilidad de usuario de pagos [0-1] (default: ${DEFAULTS.paymentProb})
  --replications <n>      Réplicas iniciales (default: ${DEFAULTS.replications})
  --relative-error <n>    Error relativo objetivo (default: ${DEFAULTS.relativeError})
  --wait-threshold <n>    Umbral de espera en minutos (default: ${DEFAULTS.waitThreshold})
  --max-replications <n>  Límite máximo de réplicas (default: ${DEFAULTS.maxReplications})
  --seed <n>              Semilla para resultados reproducibles
  -h, --help              Mostrar esta ayuda
`;
}

function createRng(seed) {
  if (seed === null || seed === undefined || Number.isNaN(seed)) {
    return Math.random;
  }
  let state = Math.floor(seed) % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function exponential(rate, rand) {
  const u = Math.max(rand(), Number.EPSILON);
  return -Math.log(u) / rate;
}

function parseArgs(argv) {
  const config = { ...DEFAULTS };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "-h" || arg === "--help") {
      console.log(usage());
      process.exit(0);
    }
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2);
    if (!(key in KEY_MAP)) {
      throw new Error(`Opción desconocida: ${arg}`);
    }
    const value = argv[i + 1];
    if (value === undefined) {
      throw new Error(`Falta valor para: ${arg}`);
    }
    const mappedKey = KEY_MAP[key];
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      throw new Error(`Valor inválido para ${arg}: ${value}`);
    }
    config[mappedKey] = parsed;
    i += 1;
  }
  return config;
}

function validateConfig(config) {
  if (config.tellers < 1) throw new Error("El número de cajeros debe ser >= 1.");
  if (config.hours <= 0) throw new Error("Las horas de simulación deben ser > 0.");
  if (config.arrivalRate <= 0 || config.serviceRate <= 0) {
    throw new Error("Las tasas de llegada y servicio deben ser > 0.");
  }
  if (config.paymentProb < 0 || config.paymentProb > 1) {
    throw new Error("La probabilidad de pagos debe estar entre 0 y 1.");
  }
  if (config.replications < 2) throw new Error("Las réplicas deben ser >= 2.");
  if (config.relativeError <= 0 || config.relativeError >= 1) {
    throw new Error("El error relativo debe estar entre 0 y 1.");
  }
  if (config.waitThreshold < 0) throw new Error("El umbral de espera no puede ser negativo.");
  if (config.maxReplications < config.replications) {
    throw new Error("El límite máximo de réplicas debe ser >= réplicas iniciales.");
  }
}

function simulateTeller({ hours, arrivalRate, serviceRate, paymentProb, rand }) {
  const stats = {
    totalCustomers: 0,
    totalServiceTime: 0,
    totalWaitTime: 0,
    paymentCount: 0,
    withdrawalCount: 0,
  };

  const endTime = hours * 60;
  const arrivalRatePerMin = arrivalRate / 60;
  const serviceRatePerMin = serviceRate / 60;

  let time = 0;
  let nextArrival = exponential(arrivalRatePerMin, rand);
  let nextDeparture = Number.POSITIVE_INFINITY;
  let serverBusy = false;
  const queue = [];

  while (nextArrival <= endTime || serverBusy || queue.length > 0) {
    const arrivalEvent = nextArrival <= endTime && nextArrival <= nextDeparture;

    if (arrivalEvent) {
      time = nextArrival;
      const isPayment = rand() < paymentProb;
      const serviceTime = exponential(serviceRatePerMin, rand);
      if (!serverBusy) {
        serverBusy = true;
        nextDeparture = time + serviceTime;
        stats.totalServiceTime += serviceTime;
        stats.totalCustomers += 1;
        if (isPayment) stats.paymentCount += 1;
        else stats.withdrawalCount += 1;
      } else {
        queue.push({ arrivalTime: time, serviceTime, isPayment });
      }
      nextArrival = time + exponential(arrivalRatePerMin, rand);
    } else {
      time = nextDeparture;
      if (queue.length > 0) {
        const nextCustomer = queue.shift();
        stats.totalWaitTime += time - nextCustomer.arrivalTime;
        stats.totalServiceTime += nextCustomer.serviceTime;
        stats.totalCustomers += 1;
        if (nextCustomer.isPayment) stats.paymentCount += 1;
        else stats.withdrawalCount += 1;
        nextDeparture = time + nextCustomer.serviceTime;
      } else {
        serverBusy = false;
        nextDeparture = Number.POSITIVE_INFINITY;
        if (nextArrival > endTime) break;
      }
    }
  }

  return stats;
}

function average(values) {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function stdev(values) {
  if (values.length < 2) return 0;
  const mean = average(values);
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

function computeRequiredReplications(samples, relativeError) {
  const mean = average(samples);
  const sigma = stdev(samples);
  if (samples.length < 2 || mean === 0 || sigma === 0) {
    return { required: samples.length, mean, sigma };
  }
  const z = Z_SCORE_95_CI;
  const halfWidthTarget = Math.abs(relativeError * mean);
  const required = Math.ceil((z * sigma / halfWidthTarget) ** 2);
  return { required: Math.max(required, samples.length), mean, sigma };
}

function simulateBank(config) {
  const aggregated = Array.from({ length: config.tellers }, () => ({
    totalCustomers: 0,
    totalServiceTime: 0,
    totalWaitTime: 0,
    paymentCount: 0,
    withdrawalCount: 0,
  }));
  const meanWaitSamples = [];
  const rand = createRng(config.seed);

  const runReplication = () => {
    let replicationWait = 0;
    let replicationCustomers = 0;
    for (let tellerIndex = 0; tellerIndex < config.tellers; tellerIndex += 1) {
      const stats = simulateTeller({
        hours: config.hours,
        arrivalRate: config.arrivalRate,
        serviceRate: config.serviceRate,
        paymentProb: config.paymentProb,
        rand,
      });
      const aggregate = aggregated[tellerIndex];
      aggregate.totalCustomers += stats.totalCustomers;
      aggregate.totalServiceTime += stats.totalServiceTime;
      aggregate.totalWaitTime += stats.totalWaitTime;
      aggregate.paymentCount += stats.paymentCount;
      aggregate.withdrawalCount += stats.withdrawalCount;
      replicationWait += stats.totalWaitTime;
      replicationCustomers += stats.totalCustomers;
    }
    meanWaitSamples.push(replicationCustomers === 0 ? 0 : replicationWait / replicationCustomers);
  };

  let required = config.replications;
  let capped = false;

  while (meanWaitSamples.length < required) {
    runReplication();
  }

  let evaluation = computeRequiredReplications(meanWaitSamples, config.relativeError);
  while (evaluation.required > meanWaitSamples.length && !capped) {
    required = evaluation.required;
    if (required > config.maxReplications) {
      required = config.maxReplications;
      capped = true;
    }
    while (meanWaitSamples.length < required) {
      runReplication();
    }
    evaluation = computeRequiredReplications(meanWaitSamples, config.relativeError);
  }

  return {
    aggregated,
    meanWaitSamples,
    replicationsExecuted: meanWaitSamples.length,
    requiredReplications: capped ? required : evaluation.required,
    capped,
  };
}

function formatMinutes(value) {
  return `${value.toFixed(2)} min`;
}

function recommendExclusiveTellers(totalTellers, totalPayments, totalWithdrawals) {
  if (totalTellers <= 0) {
    return { paymentTellers: 0, withdrawalTellers: 0 };
  }

  const totalCustomers = totalPayments + totalWithdrawals;
  const paymentShare = totalCustomers === 0 ? 0.5 : totalPayments / totalCustomers;
  let paymentTellers = Math.round(totalTellers * paymentShare);
  let withdrawalTellers = totalTellers - paymentTellers;

  if (totalTellers === 1) {
    if (totalPayments >= totalWithdrawals) {
      paymentTellers = 1;
      withdrawalTellers = 0;
    } else {
      paymentTellers = 0;
      withdrawalTellers = 1;
    }
    return { paymentTellers, withdrawalTellers };
  }

  if (paymentTellers === 0 && totalPayments > 0) paymentTellers = 1;
  if (paymentTellers >= totalTellers) paymentTellers = totalTellers - 1;
  withdrawalTellers = totalTellers - paymentTellers;
  if (withdrawalTellers === 0 && totalWithdrawals > 0) {
    withdrawalTellers = 1;
    paymentTellers = totalTellers - 1;
  }

  return { paymentTellers, withdrawalTellers };
}

function main() {
  try {
    const config = parseArgs(process.argv.slice(2));
    validateConfig(config);

    if (config.arrivalRate >= config.serviceRate) {
      console.log(
        "Advertencia: la tasa de llegada es >= la tasa de servicio; el sistema puede ser inestable."
      );
    }

    const results = simulateBank(config);
    const tellerAverages = results.aggregated.map((stats) => ({
      avgService: stats.totalCustomers === 0 ? 0 : stats.totalServiceTime / stats.totalCustomers,
      avgWait: stats.totalCustomers === 0 ? 0 : stats.totalWaitTime / stats.totalCustomers,
      customers: stats.totalCustomers,
      paymentCount: stats.paymentCount,
      withdrawalCount: stats.withdrawalCount,
    }));

    const totalCustomers = tellerAverages.reduce((sum, teller) => sum + teller.customers, 0);
    const totalPayments = tellerAverages.reduce((sum, teller) => sum + teller.paymentCount, 0);
    const totalWithdrawals = tellerAverages.reduce(
      (sum, teller) => sum + teller.withdrawalCount,
      0
    );
    const totalWait = results.aggregated.reduce((sum, teller) => sum + teller.totalWaitTime, 0);

    const avgPaymentsPerTeller = totalPayments / config.tellers;
    const avgWithdrawalsPerTeller = totalWithdrawals / config.tellers;
    const overallAvgWait = totalCustomers === 0 ? 0 : totalWait / totalCustomers;

    let minIndex = 0;
    let maxIndex = 0;
    for (let i = 1; i < tellerAverages.length; i += 1) {
      if (tellerAverages[i].avgService < tellerAverages[minIndex].avgService) minIndex = i;
      if (tellerAverages[i].avgService > tellerAverages[maxIndex].avgService) maxIndex = i;
    }

    const needsNewTeller = overallAvgWait > config.waitThreshold;
    const totalTellersRecommended = config.tellers + (needsNewTeller ? 1 : 0);
    const { paymentTellers, withdrawalTellers } = recommendExclusiveTellers(
      totalTellersRecommended,
      totalPayments,
      totalWithdrawals
    );

    console.log("Resultados de simulación M/M/1");
    console.log(`Réplicas ejecutadas: ${results.replicationsExecuted}`);
    console.log(
      `Réplicas recomendadas (95% CI, error relativo ${(
        config.relativeError * 100
      ).toFixed(1)}%): ${results.requiredReplications}${results.capped ? " (limitado)" : ""}`
    );
    console.log("");
    console.log("Promedio de tiempos por cajero:");
    tellerAverages.forEach((teller, index) => {
      console.log(
        `  Cajero ${index + 1}: atención ${formatMinutes(teller.avgService)}, espera ${formatMinutes(
          teller.avgWait
        )}, usuarios ${teller.customers}`
      );
    });
    console.log("");
    console.log(
      `Cajero con menor tiempo promedio de atención: Cajero ${minIndex + 1} (${formatMinutes(
        tellerAverages[minIndex].avgService
      )})`
    );
    console.log(
      `Cajero con mayor tiempo promedio de atención: Cajero ${maxIndex + 1} (${formatMinutes(
        tellerAverages[maxIndex].avgService
      )})`
    );
    console.log("");
    console.log(
      `Promedio de usuarios por tipo (por cajero): pagos ${avgPaymentsPerTeller.toFixed(
        2
      )}, retiros ${avgWithdrawalsPerTeller.toFixed(2)}`
    );
    console.log(`Total de usuarios: pagos ${totalPayments}, retiros ${totalWithdrawals}`);
    console.log("");
    console.log(
      `Tiempo promedio de espera global: ${formatMinutes(overallAvgWait)} (umbral ${
        config.waitThreshold
      } min)`
    );
    console.log(
      needsNewTeller
        ? "Conclusión: se recomienda crear un nuevo cajero."
        : "Conclusión: no se requiere un nuevo cajero."
    );
    console.log(
      `Cajeros exclusivos recomendados: pagos ${paymentTellers}, retiros ${withdrawalTellers}`
    );
  } catch (error) {
    console.error(`Error: ${error.message}`);
    console.log(usage());
    process.exit(1);
  }
}

main();
