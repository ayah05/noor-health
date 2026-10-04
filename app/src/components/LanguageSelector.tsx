interface Props {
  value: string;
  onChange: (language: string) => void;
}

const languages = [
  {
    code: "en",
    label: "English",
  },
  {
    code: "de",
    label: "Deutsch",
  },
  {
    code: "ar_msa",
    label: "العربية",
  },
  {
    code: "fr",
    label: "Français",
  },
  {
    code: "es",
    label: "Español",
  },
  {
    code: "hi",
    label: "हिन्दी",
  },
  {
    code: "sw",
    label: "Kiswahili",
  },
];

export default function LanguageSelector({
  value,
  onChange,
}: Props) {
  return (
    <div className="field">
      <label htmlFor="language">
        Patient language
      </label>

      <select
        id="language"
        value={value}
        onChange={(event) =>
          onChange(event.target.value)
        }
      >
        {languages.map((language) => (
          <option
            key={language.code}
            value={language.code}
          >
            {language.label}
          </option>
        ))}
      </select>
    </div>
  );
}