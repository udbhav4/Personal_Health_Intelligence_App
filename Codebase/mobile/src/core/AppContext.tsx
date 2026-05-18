import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useRef,
  type ReactNode,
} from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import type { DB } from '@op-engineering/op-sqlite';
import * as FileSystem from 'expo-file-system/legacy';

import { openDb, initDb }       from './db';
import { setModelPaths }         from './initModels';
import type { ModelPaths }       from './initModels';
import { initPassiveSensing }    from './passiveSensing';
import type { BeliefResult }     from './inferenceEngine';
import {
  computeBmi, discretizeBmi,
  computeAgeYears, discretizeAge,
}                                from './onboardingHelpers';

// ── Constants ─────────────────────────────────────────────────────────────────

const STORAGE_KEY    = 'gliimr_onboarding';
const COMPLETED_KEY  = 'gliimr_completed';
const MODEL_PATHS_KEY = 'gliimr_model_paths';

export interface UserProfile {
  name:      string;
  sex:       string;
  birthdate: string;
  weight:    string;
  height:    string;
}

function makeSessionId(): string {
  return `sess-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

// ── Context shape ─────────────────────────────────────────────────────────────

interface AppContextValue {
  db:                     DB | null;
  sessionId:              string;
  modelsReady:            boolean;
  modelsDownloaded:       boolean;
  appReady:               boolean;
  initError:              string | null;
  beliefs:                BeliefResult | null;
  setBeliefs:             (b: BeliefResult | null) => void;
  hasCompletedOnboarding: boolean | null;
  profile:                UserProfile;
  saveProfile:            (data: UserProfile) => Promise<void>;
  updateProfile:          (data: UserProfile) => Promise<void>;
  onModelsReady:          (paths: ModelPaths) => Promise<void>;
}

const AppContext = createContext<AppContextValue | null>(null);

export function useAppContext(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useAppContext must be used inside AppProvider');
  return ctx;
}

// ── Provider ──────────────────────────────────────────────────────────────────

export function AppProvider({ children }: { children: ReactNode }) {
  const [db,          setDb]          = useState<DB | null>(null);
  const _dbRef                        = useRef<DB | null>(null);
  const [sessionId]                   = useState(makeSessionId);
  const [modelsReady,      setModelsReady]      = useState(false);
  const [modelsDownloaded, setModelsDownloaded] = useState(false);
  const [appReady,         setAppReady]         = useState(false);
  const [initError,   setInitError]   = useState<string | null>(null);
  const [beliefs,     setBeliefs]     = useState<BeliefResult | null>(null);

  const [hasCompletedOnboarding, setHasCompletedOnboarding] = useState<boolean | null>(null);
  const [profile, setProfile] = useState<UserProfile>({
    name: '', sex: '', birthdate: '', weight: '', height: '',
  });

  const initialized = useRef(false);

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;

    (async () => {
      try {
        const database = openDb();
        initDb(database);
        // ── DEMO SEED ── remove this entire block to ship without demo data ──────
        if (__DEV__) {
          try {
            const { seedMiddleAgedData } = require('./seedDb_middleAged') as typeof import('./seedDb_middleAged');
            seedMiddleAgedData(database);
          } catch { /* seed failure is non-fatal */ }
        }
        // ── END DEMO SEED ────────────────────────────────────────────────────────
        _dbRef.current = database;
        setDb(database);

        initPassiveSensing(database).catch(() => {});

        // Pre-populate beliefs from latest snapshot so dashboard rings show on first open
        try {
          const snapRow = database.executeSync(
            `SELECT dbn_beliefs FROM inference_snapshots ORDER BY date DESC, snapshot_time DESC LIMIT 1`,
          ).rows[0] as { dbn_beliefs: string } | undefined;
          if (snapRow?.dbn_beliefs) setBeliefs(JSON.parse(snapRow.dbn_beliefs));
        } catch {}

        const [completedStr, savedProfileStr] = await Promise.all([
          AsyncStorage.getItem(COMPLETED_KEY),
          AsyncStorage.getItem(STORAGE_KEY),
        ]);

        if (savedProfileStr) {
          try { setProfile(JSON.parse(savedProfileStr)); } catch {}
        }
        setHasCompletedOnboarding(completedStr === 'true');

        const savedPathsStr = await AsyncStorage.getItem(MODEL_PATHS_KEY);
        if (savedPathsStr) {
          try {
            const paths = JSON.parse(savedPathsStr) as ModelPaths;
            const [a, b, c] = await Promise.all([
              FileSystem.getInfoAsync(paths.nlu),
              FileSystem.getInfoAsync(paths.embed),
              FileSystem.getInfoAsync(paths.agent),
            ]);
            if (a.exists && b.exists && c.exists) {
              setModelPaths(paths);
              setModelsDownloaded(true);
              setModelsReady(true);
            }
          } catch { /* corrupted paths — fall through to download screen */ }
        }
      } catch (e) {
        setInitError(String(e));
      } finally {
        setAppReady(true);
      }
    })();
  }, []);

  const saveProfile = async (data: UserProfile) => {
    await AsyncStorage.multiSet([
      [STORAGE_KEY,   JSON.stringify(data)],
      [COMPLETED_KEY, 'true'],
    ]);
    setProfile(data);
    setHasCompletedOnboarding(true);

    // Persist BMI and age to user_data_sensorless as onboarding evidence
    const database = _dbRef.current;
    if (database) {
      const ts = new Date().toISOString().replace('T', ' ').slice(0, 19);
      const today = new Date().toISOString().slice(0, 10);
      const turnId = `onboard-${Date.now().toString(36)}`;

      // Parse weight (e.g. "68 kg" → 68) and height (e.g. "175 cm" → 175)
      const weightKg = parseFloat(data.weight);
      const heightCm = parseFloat(data.height);

      if (!isNaN(weightKg) && !isNaN(heightCm) && heightCm > 0) {
        const bmi     = computeBmi(weightKg, heightCm);
        const bmiState = discretizeBmi(bmi);
        try {
          database.executeSync(
            `INSERT INTO user_data_sensorless
               (timestamp, node_name, raw_text, raw_value, node_value,
                data_source, merge_mode, temporal_flag, report_date, turn_id, answered)
             VALUES (?, 'bmi', ?, ?, ?, 'onboarding', 'latest', 'persistent', ?, ?, 1)`,
            [ts, `${bmi.toFixed(1)}`, bmi, bmiState, today, turnId],
          );
        } catch { /* schema may not be ready yet — ignore */ }
      }

      // Parse birthdate DD-Mon-YYYY → ISO YYYY-MM-DD
      if (data.birthdate) {
        const MONTH_MAP: Record<string, string> = {
          Jan:'01',Feb:'02',Mar:'03',Apr:'04',May:'05',Jun:'06',
          Jul:'07',Aug:'08',Sep:'09',Oct:'10',Nov:'11',Dec:'12',
        };
        const parts = data.birthdate.split('-');
        if (parts.length === 3) {
          const isoDate = `${parts[2]}-${MONTH_MAP[parts[1]] ?? parts[1]}-${parts[0]}`;
          const ageYears = computeAgeYears(isoDate);
          const ageState = discretizeAge(ageYears);
          try {
            database.executeSync(
              `INSERT INTO user_data_sensorless
                 (timestamp, node_name, raw_text, raw_value, node_value,
                  data_source, merge_mode, temporal_flag, report_date, turn_id, answered)
               VALUES (?, 'age', ?, ?, ?, 'onboarding', 'latest', 'persistent', ?, ?, 1)`,
              [ts, isoDate, ageYears, ageState, today, turnId],
            );
          } catch { /* ignore */ }
        }
      }

      // Write sex to user_data_sensorless ('Male'/'Female' → lowercase for DBN)
      if (data.sex) {
        const sexState = data.sex.toLowerCase() as 'male' | 'female';
        try {
          database.executeSync(
            `INSERT INTO user_data_sensorless
               (timestamp, node_name, raw_text, raw_value, node_value,
                data_source, merge_mode, temporal_flag, report_date, turn_id, answered)
             VALUES (?, 'sex', ?, NULL, ?, 'onboarding', 'latest', 'persistent', ?, ?, 1)`,
            [ts, data.sex, sexState, today, turnId],
          );
        } catch { /* ignore */ }
      }
    }
  };

  const updateProfile = async (data: UserProfile) => {
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    setProfile(data);
  };

  const onModelsReady = async (paths: ModelPaths): Promise<void> => {
    await AsyncStorage.setItem(MODEL_PATHS_KEY, JSON.stringify(paths));
    setModelPaths(paths);
    setModelsReady(true);
    setModelsDownloaded(true);
  };

  return (
    <AppContext.Provider value={{
      db, sessionId, modelsReady, modelsDownloaded, appReady, initError,
      beliefs, setBeliefs,
      hasCompletedOnboarding, profile,
      saveProfile, updateProfile,
      onModelsReady,
    }}>
      {children}
    </AppContext.Provider>
  );
}
